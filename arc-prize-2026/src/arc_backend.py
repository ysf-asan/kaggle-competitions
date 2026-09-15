"""Model backend, so the notebook does not depend on unsloth being present.

unsloth is not part of the Kaggle image. It comes from a utility script pinned
to a Python 3.11 image, and attaching that script to a notebook on a newer
image makes the session fail to start at all. That single dependency has been
the only thing standing between us and a run, so it is now optional:

- unsloth importable  -> fast path, byte-for-byte the behaviour of the public
  baseline (patched attention, unsloth's trainer).
- unsloth missing     -> plain transformers + peft. Same model, same LoRA
  config, same schedule, same decoding. Slower per task, which the two-pass
  budget absorbs by revisiting fewer tasks; nothing about the method changes.

Everything the solver needs from a backend goes through this module.
"""

import importlib.util
import torch

import arc_config


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_UNSLOTH = _importable("unsloth")

LORA = dict(
    r=arc_config.LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_rslora=True,
)


def name():
    return "unsloth" if HAS_UNSLOTH else "transformers"


def load_model(model_path, max_seq_length, compute_dtype, pad_token_id):
    if HAS_UNSLOTH:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=False,
            max_seq_length=max_seq_length,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, dtype=compute_dtype, local_files_only=True,
                attn_implementation="sdpa",
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=compute_dtype, local_files_only=True,
                attn_implementation="sdpa",
            )
        model = model.cuda()

    if tokenizer.pad_token_id != pad_token_id:
        print(f"*** pad_token_id {tokenizer.pad_token_id} -> {pad_token_id}")
        tokenizer.pad_token_id = pad_token_id
    return model, tokenizer


def neutralise_optional_backend_probes():
    """Stop peft's quantiser probes from raising on a plain LoRA injection.

    peft's LoRA dispatcher asks each optional backend whether it is available.
    `is_torchao_available` does not answer False when the installed torchao is
    older than peft wants — it raises ImportError. The Kaggle image ships
    torchao 0.10 against a peft that wants >0.16, so injecting an ordinary
    bf16 LoRA adapter dies on a quantiser we never use. Answer False for it.
    """
    patched = []
    try:
        from peft import import_utils
    except ImportError:
        return patched

    probe = getattr(import_utils, "is_torchao_available", None)
    if probe is None:
        return patched
    try:
        probe()
        return patched
    except ImportError:
        pass

    false = lambda: False
    import_utils.is_torchao_available = false
    patched.append("peft.import_utils")

    try:
        from peft.tuners.lora import torchao as lora_torchao
    except ImportError:
        return patched
    if hasattr(lora_torchao, "is_torchao_available"):
        lora_torchao.is_torchao_available = false
        patched.append("peft.tuners.lora.torchao")
    return patched


def get_peft_model(model):
    if HAS_UNSLOTH:
        from unsloth import FastLanguageModel
        return FastLanguageModel.get_peft_model(
            model, use_gradient_checkpointing=False, random_state=42,
            loftq_config=None, **LORA,
        )
    patched = neutralise_optional_backend_probes()
    if patched:
        print(f"*** disabled torchao probe in: {', '.join(patched)}")
    from peft import LoraConfig, get_peft_model as peft_get_peft_model
    return peft_get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))


def for_training(model):
    if HAS_UNSLOTH:
        from unsloth import FastLanguageModel
        return FastLanguageModel.for_training(model)
    model.train()
    return model


def for_inference(model):
    if HAS_UNSLOTH:
        from unsloth import FastLanguageModel
        return FastLanguageModel.for_inference(model)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = True

    model.eval()
    return model


def eval_strategy_key():
    """`evaluation_strategy` was renamed to `eval_strategy` in transformers
    4.46, and passing the name this version does not know raises."""
    try:
        import inspect
        from transformers import TrainingArguments
        fields = inspect.signature(TrainingArguments.__init__).parameters
    except Exception:
        return "eval_strategy"
    return "eval_strategy" if "eval_strategy" in fields else "evaluation_strategy"


def adjust_train_args(train_args):
    """Memory settings the plain path needs and the unsloth path does not.

    The reference runs without gradient checkpointing because unsloth's fused
    kernels keep the activations small enough. Plain transformers does not:
    thirty-six layers of an 8192-token sequence will not fit beside a rank-256
    adapter and its optimiser state on a 22 GB L4. Recomputing activations in
    the backward pass costs perhaps a third of the speed and changes nothing
    about the result.
    """
    out = dict(train_args)
    if HAS_UNSLOTH:
        return out
    out["gradient_checkpointing"] = True
    out["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return out


def build_trainer(model, tokenizer, collator, samples, train_args,
                  max_seq_length, fixed_trainer_cls=None, callbacks=None):
    """Trainer over `samples`, a list of dicts carrying a 'text' field."""
    from datasets import Dataset

    if HAS_UNSLOTH:
        from unsloth import UnslothTrainingArguments
        return fixed_trainer_cls(
            model=model,
            tokenizer=tokenizer,
            data_collator=collator,
            train_dataset=Dataset.from_list(samples),
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=UnslothTrainingArguments(**train_args),
            callbacks=callbacks,
        )

    from transformers import Trainer, TrainingArguments

    dataset = Dataset.from_list(samples)
    columns = dataset.column_names

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)

    dataset = dataset.map(tokenize, batched=True, remove_columns=columns)

    train_args = adjust_train_args(train_args)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    args = TrainingArguments(output_dir="/tmp/arc_trainer", **train_args)
    try:
        return Trainer(model=model, args=args, data_collator=collator,
                       train_dataset=dataset, processing_class=tokenizer,
                       callbacks=callbacks)
    except TypeError:
        return Trainer(model=model, args=args, data_collator=collator,
                       train_dataset=dataset, tokenizer=tokenizer,
                       callbacks=callbacks)


def unwrap(trainer, model):
    try:
        return trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    except (AttributeError, TypeError):
        return model


_cache_reported = False


def prepare_cache(cache, pos):
    """Return a KV cache holding exactly `pos` tokens.

    The DFS explores sibling continuations of the same prefix, so every forward
    at a given depth must start from that prefix. Legacy tuple caches are
    immutable — each forward returns a fresh tuple and the parent's cache is
    untouched — but a modern `Cache` object is mutated in place and handed
    back, so without cropping, sibling branches would each extend the previous
    one's cache and the search would silently decode nonsense.
    """
    global _cache_reported
    crop = getattr(cache, "crop", None)
    if not _cache_reported:
        _cache_reported = True
        kind = type(cache).__name__
        print(f"*** kv cache: {kind}, "
              + ("croppable" if crop is not None
                 else "no crop() — assumed immutable per forward"))
    if crop is not None:
        try:
            crop(pos)
        except (AttributeError, TypeError, IndexError):
            pass
    return cache
