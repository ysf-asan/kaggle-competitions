import arc_backend
from arc_loader import ArcDataset, QwenFormatter
import arc_config
from arc_batching import build_decode_batches
from arc_labels import build_completion_labels, marker_sequences

if arc_backend.HAS_UNSLOTH:
    from unsloth import UnslothTrainingArguments, UnslothTrainer
else:
    UnslothTrainer = object

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling, TrainerCallback

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class DeadlineCallback(TrainerCallback):
    """Stop test-time training when its share of the task budget is spent.

    Training a long task to completion can consume the whole per-task cap, and
    the decoder then never runs: the task costs full price and returns nothing.
    Cutting training short leaves a less adapted model but one that still gets
    to answer.
    """

    def __init__(self, deadline):
        self.deadline = deadline
        self.stopped_early = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            self.stopped_early = True
            control.should_training_stop = True
        return control


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    assistant_header = None
    user_header = None
    end_marker = None

    supervised = None
    total = None

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = build_completion_labels(
                batch["input_ids"][i].tolist(),
                self.assistant_header,
                self.end_marker,
                user_header=self.user_header,
            )
            batch["labels"][i] = torch.tensor(
                labels, dtype=batch["labels"].dtype, device=batch["labels"].device)

        if QwenDataCollatorForCompletionOnlyLM.supervised is None:
            QwenDataCollatorForCompletionOnlyLM.supervised = int((batch["labels"] != -100).sum())
            QwenDataCollatorForCompletionOnlyLM.total = int(batch["labels"].numel())
        return batch


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, dfs_cap=540) -> dict:

    n = logits.size(0)

    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])
    
    while time.time() - start_time < dfs_cap and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=arc_backend.prepare_cache(cache, pos),
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            dfs_cap=dfs_cap,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, dfs_cap=540):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    if outputs.past_key_values is None:
        raise RuntimeError(
            "model returned no KV cache; decoding needs use_cache to be honoured"
        )
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        dfs_cap=dfs_cap,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def worker(rank, queue, end_time, pass_id=1, n_workers=1):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    budget = arc_config.PASS_BUDGETS[pass_id]

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(f"[Rank {rank}] compute dtype: {compute_dtype}")


    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=budget.ttt_epochs,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
        fp16=not supports_bf16,
        bf16=supports_bf16,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    train_args[arc_backend.eval_strategy_key()] = "no"

    max_seq_length = arc_config.MAX_SEQ_LENGTH

    model_path = arc_config.resolve_model_path()
    if model_path is None:
        raise FileNotFoundError(
            f"No model checkpoint under {arc_config.INPUT_ROOT}. Expected "
            f"{arc_config.MODEL_PATH}; attach "
            f"{arc_config.MODEL_OWNER}/{arc_config.MODEL_SLUG}."
        )
    print(f"[Rank {rank}] model: {model_path}")

    model, tokenizer = arc_backend.load_model(
        model_path, max_seq_length, compute_dtype, PAD_ID)

    model = arc_backend.get_peft_model(model)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(compute_dtype)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    markers = marker_sequences(tokenizer)
    collator.assistant_header = markers["assistant"]
    collator.user_header = markers["user"]
    collator.end_marker = markers["end"]
    print(f"[Rank {rank}] markers: user={markers['user']} "
          f"assistant={markers['assistant']} end={markers['end']}")
    if not markers["assistant"] or not markers["end"]:
        raise RuntimeError(
            "tokenizer does not produce the chat markers the formatter writes; "
            "test-time training would have nothing to supervise"
        )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = budget.max_score

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    tokens_per_second = arc_config.TOKENS_PER_SECOND_PRIOR
    reported = False

    dir_outputs = arc_config.OUTPUT_DIRS[pass_id]
    os.makedirs(dir_outputs, exist_ok=True)

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()

        if budget.adaptive_cap:
            try:
                remaining_tasks = max(1.0, queue.qsize() / n_workers + 1)
            except NotImplementedError:
                remaining_tasks = 1.0
            task_cap = min(budget.per_task_cap,
                           max(120.0, (end_time - start_time) / remaining_tasks))
        else:
            task_cap = budget.per_task_cap
        
        try:
            torch.cuda.reset_peak_memory_stats()

            load_result = set_peft_model_state_dict(
                model,
                default_weights.copy(),
                adapter_name="default",
            )

            model = arc_backend.for_training(model)

            deadline = DeadlineCallback(min(
                start_time + task_cap * arc_config.TTT_TIME_FRACTION,
                end_time,
            ))

            puzzle_ds = arc_test_set.change_keys([key])

            train_ds = puzzle_ds.augment(n=budget.ttt_augmentations, shfl_keys=True, seed=1)
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)
            samples = train_ds.as_list(formatter)

            task_tokens = max(1, len(tokenizer.encode(samples[0]["text"])))
            step_seconds = task_tokens / tokens_per_second

            task_train_args = dict(train_args)
            affordable = int(task_cap * arc_config.TTT_TIME_FRACTION / step_seconds)
            planned = int(np.clip(affordable, 8, len(samples) * budget.ttt_epochs))
            task_train_args["max_steps"] = planned
            task_train_args.pop("num_train_epochs", None)

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            
                trainer = arc_backend.build_trainer(
                    model=model,
                    tokenizer=tokenizer,
                    collator=collator,
                    samples=samples,
                    train_args=task_train_args,
                    max_seq_length=max_seq_length,
                    fixed_trainer_cls=UnslothFixedTrainer,
                    callbacks=[deadline],
                )

                stats = trainer.train()

                model = arc_backend.unwrap(trainer, model)

                del trainer

            model = arc_backend.for_inference(model)
        
            gc.collect()
            torch.cuda.empty_cache()
            
            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

            torch.cuda.reset_peak_memory_stats()
        
            if stats.global_step and stats.metrics.get("train_runtime"):
                measured = task_tokens * stats.global_step / stats.metrics["train_runtime"]
                tokens_per_second = 0.7 * tokens_per_second + 0.3 * measured
                print(f"[Rank {rank}] planned {planned} steps, ran "
                      f"{stats.global_step}; throughput {measured:.0f} tok/s")

            if QwenDataCollatorForCompletionOnlyLM.supervised is not None and not reported:
                reported = True
                sup = QwenDataCollatorForCompletionOnlyLM.supervised
                tot = QwenDataCollatorForCompletionOnlyLM.total
                print(f"[Rank {rank}] collator: {sup}/{tot} supervised label positions")
                if sup == 0:
                    print(f"[Rank {rank}] WARNING: nothing supervised — "
                          f"test-time training cannot learn")
                current = get_peft_model_state_dict(model, adapter_name="default")
                delta = max(
                    (current[k].float() - default_weights[k].float()).norm().item()
                    for k in list(current)[:8]
                )
                print(f"[Rank {rank}] adapter delta after training: {delta:.3e}")
                if delta == 0.0:
                    print(f"[Rank {rank}] WARNING: adapter unchanged — "
                          f"test-time training is a no-op")

            if deadline.stopped_early:
                print(f"[Rank {rank}] training hit its deadline")
            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

            puzzle_ds_multi = puzzle_ds.split_multi_replies()

            eval_ds = puzzle_ds_multi.augment(n=budget.eval_permutations, seed=2)
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

            test_id_to_subkeys = defaultdict(list)
            for subkey in sorted(eval_ds.keys):
                test_id = subkey.split(".")[0].split("_")[1]
                test_id_to_subkeys[test_id].append(subkey)

            batches = build_decode_batches(test_id_to_subkeys, budget.eval_permutations)

            with torch.inference_mode():
                
                known_scores = {}

                for subkeys in batches:

                    spend_time = time.time() - start_time
                    if spend_time > task_cap or time.time() > end_time:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")

                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time, dfs_cap=budget.dfs_cap)

                    for subkey_id, scored_beams in dfs_result:

                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []

                        for beam_score, tokens in scored_beams:

                            array = formatter.convert_tokens_to_array(tokens)
                            if array is None:
                                continue

                            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                            grid_id = (bk, tuple(map(tuple, solution)))

                            if grid_id in known_scores:
                                augmented_scores = known_scores[grid_id]
                            else:
                                print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                                aug_dataset = ArcDataset(
                                    keys=[bk],
                                    queries={bk: puzzle_ds_multi.queries.get(bk)},
                                    replies={bk: [solution.tolist()]},
                                )
                                aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                                aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                                augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                                augmented_scores = augmented_scores1 + augmented_scores2
                                known_scores[grid_id] = augmented_scores
                        
                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            })

                        if len(decoded_result):
                            with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                                pickle.dump(decoded_result, f)

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        
            spend_time = time.time() - start_time
            print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")

        except torch.OutOfMemoryError:
            print(f"[Rank {rank}] OUT OF MEMORY on {key}, skipping it")
        except Exception as e:
            print(f"[Rank {rank}] FAILED on {key}: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
