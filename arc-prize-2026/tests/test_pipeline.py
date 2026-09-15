"""Off-GPU tests for the parts of the pipeline that are pure Python.

The selection rules, the confidence ranking and the decode batching decide how
compute is spent and which grid ends up as attempt_1, and all three can be
exercised without a model. Run with: python -m pytest tests -q
"""

import os
import sys
import bz2
import pickle
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import preflight
from arc_batching import build_decode_batches
from arc_confidence import output_confidence, select_pass2_queue, rank_tasks
from arc_decoder import (
    infer_shape_rule, expected_shape, apply_shape_prior,
    score_kgmon, ArcDecoder,
)


def pair(inp, out):
    return {"input": inp, "output": out}


def grid(h, w, fill=0):
    return [[fill] * w for _ in range(h)]


def test_rule_same_shape():
    train = [pair(grid(3, 3), grid(3, 3)), pair(grid(5, 2), grid(5, 2))]
    assert infer_shape_rule(train) == ("same", None)
    assert expected_shape(("same", None), (7, 4)) == (7, 4)


def test_rule_constant_output():
    train = [pair(grid(3, 3), grid(1, 1)), pair(grid(5, 2), grid(1, 1))]
    assert infer_shape_rule(train) == ("const", (1, 1))
    assert expected_shape(("const", (1, 1)), (9, 9)) == (1, 1)


def test_rule_integer_upscale():
    train = [pair(grid(2, 3), grid(4, 6)), pair(grid(3, 1), grid(6, 2))]
    assert infer_shape_rule(train) == ("scale", (2, 2))
    assert expected_shape(("scale", (2, 2)), (5, 5)) == (10, 10)


def test_rule_integer_downscale():
    train = [pair(grid(6, 9), grid(2, 3)), pair(grid(3, 6), grid(1, 2))]
    assert infer_shape_rule(train) == ("div", (3, 3))
    assert expected_shape(("div", (3, 3)), (9, 9)) == (3, 3)
    assert expected_shape(("div", (3, 3)), (7, 9)) is None


def test_no_rule_when_shapes_are_unrelated():
    train = [pair(grid(3, 3), grid(2, 5)), pair(grid(4, 4), grid(7, 1))]
    assert infer_shape_rule(train) is None


def test_single_pair_is_never_enough():
    assert infer_shape_rule([pair(grid(3, 3), grid(3, 3))]) is None


def query_with(train, test_input):
    return {"train": train, "test": [{"input": test_input}]}


def test_prior_promotes_conforming_candidate():
    q = query_with([pair(grid(3, 3), grid(3, 3)), pair(grid(4, 4), grid(4, 4))],
                   grid(5, 5))
    wrong = np.zeros((4, 5), dtype=int)
    right = np.ones((5, 5), dtype=int)
    out = apply_shape_prior([wrong, right], q)
    assert np.array_equal(out[0], right)
    assert len(out) == 2


def test_prior_never_drops_candidates():
    q = query_with([pair(grid(3, 3), grid(3, 3)), pair(grid(4, 4), grid(4, 4))],
                   grid(5, 5))
    cands = [np.zeros((4, 5), dtype=int), np.ones((5, 5), dtype=int),
             np.zeros((2, 2), dtype=int)]
    out = apply_shape_prior(cands, q)
    assert len(out) == len(cands)
    assert {c.shape for c in out} == {c.shape for c in cands}


def test_prior_is_noop_when_nothing_conforms():
    q = query_with([pair(grid(3, 3), grid(3, 3)), pair(grid(4, 4), grid(4, 4))],
                   grid(5, 5))
    cands = [np.zeros((4, 5), dtype=int), np.zeros((2, 2), dtype=int)]
    out = apply_shape_prior(cands, q)
    assert [c.shape for c in out] == [c.shape for c in cands]


def test_prior_is_noop_without_a_rule():
    q = query_with([pair(grid(3, 3), grid(2, 5)), pair(grid(4, 4), grid(7, 1))],
                   grid(5, 5))
    cands = [np.zeros((4, 5), dtype=int), np.ones((5, 5), dtype=int)]
    out = apply_shape_prior(cands, q)
    assert np.array_equal(out[0], cands[0])


def guess(solution, beam_score=1.0, aug=None):
    return {"solution": np.array(solution), "beam_score": beam_score,
            "score_aug": aug if aug is not None else [1.0] * 8}


def test_unanimous_full_coverage_is_confident():
    guesses = [guess([[1]]) for _ in range(16)]
    assert output_confidence(guesses, expected_results=16) > 0.9


def test_starved_task_is_not_confident():
    guesses = [guess([[1]]) for _ in range(2)]
    assert output_confidence(guesses, expected_results=16) < 0.2


def test_split_vote_is_less_confident_than_unanimous():
    split = [guess([[1]]) for _ in range(8)] + [guess([[2]]) for _ in range(8)]
    unanimous = [guess([[1]]) for _ in range(16)]
    assert (output_confidence(split, 16)
            < output_confidence(unanimous, 16))


def test_no_results_is_zero():
    assert output_confidence([], expected_results=16) == 0.0


def write_store(tmp_path, per_subkey):
    store = tmp_path / "outputs"
    store.mkdir()
    for subkey, guesses in per_subkey.items():
        with bz2.BZ2File(store / subkey, "w") as f:
            pickle.dump(guesses, f)
    return str(store)


def test_unreached_tasks_come_first(tmp_path):
    store = write_store(tmp_path, {
        "aaa_0.permute0123456789": [guess([[1]]) for _ in range(16)],
    })
    queue = select_pass2_queue(["aaa", "bbb", "ccc"], store,
                               expected_results=16, fraction=0.0)
    assert queue == ["bbb", "ccc"]


def test_least_confident_reached_task_is_picked(tmp_path):
    store = write_store(tmp_path, {
        "aaa_0.permute0123456789": [guess([[1]]) for _ in range(16)],
        "bbb_0.permute0123456789": [guess([[1]]) for _ in range(2)],
    })
    queue = select_pass2_queue(["aaa", "bbb"], store,
                               expected_results=16, fraction=0.5)
    assert queue == ["bbb"]


def test_multi_output_task_takes_its_weakest_output(tmp_path):
    store = write_store(tmp_path, {
        "aaa_0.permute0123456789": [guess([[1]]) for _ in range(16)],
        "aaa_1.permute0123456789": [guess([[1]]) for _ in range(2)],
        "bbb_0.permute0123456789": [guess([[1]]) for _ in range(14)],
    })
    ranked = dict((t, c) for c, t in rank_tasks(["aaa", "bbb"], store, 16))
    assert ranked["aaa"] < ranked["bbb"]


def test_missing_store_puts_everything_in_the_queue(tmp_path):
    queue = select_pass2_queue(["aaa", "bbb"], str(tmp_path / "nope"),
                               expected_results=16, fraction=0.0)
    assert sorted(queue) == ["aaa", "bbb"]


def subkeys_for(n_perm, test_ids=("0",)):
    out = {}
    for t in test_ids:
        out[t] = [f"task_{t}.g{g}.p{p}" for g in range(8) for p in range(n_perm)]
    return out


def test_batching_matches_the_baseline_offsets():
    subkeys = subkeys_for(2)
    batches = build_decode_batches(subkeys, n_perm=2)
    flat = subkeys["0"]
    expected = [
        [flat[0], flat[1], flat[4], flat[5]],
        [flat[2], flat[3], flat[6], flat[7]],
        [flat[8], flat[9], flat[12], flat[13]],
        [flat[10], flat[11], flat[14], flat[15]],
    ]
    assert batches == expected


def test_batching_covers_every_augmentation_exactly_once():
    for n_perm in (2, 4):
        subkeys = subkeys_for(n_perm)
        batches = build_decode_batches(subkeys, n_perm=n_perm)
        seen = [k for b in batches for k in b]
        assert sorted(seen) == sorted(subkeys["0"])
        assert all(len(b) == 4 for b in batches)


def test_batching_interleaves_test_outputs():
    subkeys = subkeys_for(2, test_ids=("0", "1"))
    batches = build_decode_batches(subkeys, n_perm=2)
    assert batches[0][0].startswith("task_0")
    assert batches[1][0].startswith("task_1")


def test_batching_pairs_distant_transform_groups():
    subkeys = subkeys_for(2)
    for batch in build_decode_batches(subkeys, n_perm=2):
        groups = {k.split(".")[1] for k in batch}
        assert len(groups) == 2, "a batch must span two dihedral groups"


class FakeDataset:
    def __init__(self, queries, replies):
        self.queries = queries
        self.replies = replies


def test_decoder_merges_two_passes(tmp_path):
    store1 = write_store(tmp_path, {"aaa_0.p1": [guess([[1]])]})
    store2 = tmp_path / "outputs2"
    store2.mkdir()
    with bz2.BZ2File(store2 / "aaa_0.p1", "w") as f:
        pickle.dump([guess([[2]])], f)

    q = query_with([pair(grid(1, 1), grid(1, 1)), pair(grid(1, 1), grid(1, 1))],
                   grid(1, 1))
    decoder = ArcDecoder(FakeDataset({"aaa_0": q}, {}), n_guesses=2)
    decoder.load_decoded_results(store1, run_name="")
    decoder.load_decoded_results(str(store2), run_name="_p2")

    assert len(decoder.decoded_results["aaa_0"]) == 2
    selected = decoder.run_selection_algo(score_kgmon)
    assert len(selected["aaa_0"]) == 2


def test_worker_count_follows_the_machine(monkeypatch):
    import arc_config
    monkeypatch.setattr(arc_config, "NUM_WORKERS", None)
    assert arc_config.resolve_num_workers(4) == 4
    assert arc_config.resolve_num_workers(2) == 2
    assert arc_config.resolve_num_workers(0) == 1


def test_worker_count_override_wins(monkeypatch):
    import arc_config
    monkeypatch.setattr(arc_config, "NUM_WORKERS", 3)
    assert arc_config.resolve_num_workers(4) == 3


def make_checkpoint(root, *parts, weights="model.safetensors"):
    d = root
    for p in parts:
        d = d / p
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    (d / weights).write_bytes(b"")
    return str(d)


def test_model_path_is_used_when_it_exists(tmp_path):
    import arc_config
    expected = make_checkpoint(tmp_path, "qwen3_4b_grids15_sft139",
                               "transformers", "bfloat16", "1")
    assert arc_config.resolve_model_path(str(tmp_path), expected) == expected


def test_a_different_version_directory_is_found(tmp_path):
    """The trailing path element is the model version and it changes when the
    owner republishes; that must not take the run down."""
    import arc_config
    actual = make_checkpoint(tmp_path, "qwen3_4b_grids15_sft139",
                             "transformers", "bfloat16", "3")
    stale = str(tmp_path / "qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    assert arc_config.resolve_model_path(str(tmp_path), stale) == actual


def test_the_named_model_wins_over_other_checkpoints(tmp_path):
    import arc_config
    make_checkpoint(tmp_path, "some-other-model", "transformers", "fp16", "1")
    wanted = make_checkpoint(tmp_path, "qwen3_4b_grids15_sft139",
                             "transformers", "bfloat16", "2")
    assert arc_config.resolve_model_path(
        str(tmp_path), str(tmp_path / "nope")) == wanted


def test_a_lone_checkpoint_is_accepted(tmp_path):
    import arc_config
    only = make_checkpoint(tmp_path, "renamed-model", "transformers", "bf16", "1")
    assert arc_config.resolve_model_path(
        str(tmp_path), str(tmp_path / "nope")) == only


def test_sharded_weights_are_recognised(tmp_path):
    import arc_config
    d = make_checkpoint(tmp_path, "qwen3_4b_grids15_sft139", "transformers",
                        "bfloat16", "1", weights="model-00001-of-00002.safetensors")
    assert arc_config.resolve_model_path(str(tmp_path), d) == d


def test_a_config_without_weights_is_not_a_checkpoint(tmp_path):
    import arc_config
    d = tmp_path / "competitions" / "arc-prize-2026-arc-agi-2"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert arc_config.resolve_model_path(str(tmp_path), str(tmp_path / "nope")) is None


def test_nothing_attached_resolves_to_none(tmp_path):
    import arc_config
    assert arc_config.resolve_model_path(str(tmp_path), str(tmp_path / "nope")) is None
    assert arc_config.resolve_model_path(str(tmp_path / "missing"),
                                         str(tmp_path / "nope")) is None


def test_preflight_reports_what_is_attached_when_the_model_is_absent(tmp_path):
    import arc_config

    class FakeConfig:
        MODEL_PATH = str(tmp_path / "qwen/transformers/bfloat16/1")
        MODEL_SLUG = arc_config.MODEL_SLUG
        MODEL_OWNER = arc_config.MODEL_OWNER
        resolve_model_path = staticmethod(lambda: None)

    problem, path = preflight.find_model_problem(FakeConfig)
    assert path is None
    assert "no model checkpoint" in problem
    assert f"{arc_config.MODEL_OWNER}/{arc_config.MODEL_SLUG}" in problem


def test_preflight_passes_the_resolved_path_through(tmp_path):
    import arc_config
    found = make_checkpoint(tmp_path, "qwen3_4b_grids15_sft139",
                            "transformers", "bfloat16", "1")

    class FakeConfig:
        MODEL_PATH = found
        MODEL_SLUG = arc_config.MODEL_SLUG
        MODEL_OWNER = arc_config.MODEL_OWNER
        resolve_model_path = staticmethod(lambda: found)

    problem, path = preflight.find_model_problem(FakeConfig)
    assert problem is None and path == found


def test_describe_inputs_handles_a_missing_root(tmp_path):
    lines = preflight.describe_inputs(str(tmp_path / "nothing-here"))
    assert len(lines) == 1 and "does not exist" in lines[0]


def test_describe_inputs_lists_what_is_there(tmp_path):
    (tmp_path / "competitions").mkdir()
    (tmp_path / "qwen3_4b_grids15_sft139").mkdir()
    lines = preflight.describe_inputs(str(tmp_path))
    assert len(lines) == 2
    assert any("qwen3_4b_grids15_sft139" in l for l in lines)


def test_preflight_flags_a_missing_required_module(tmp_path):
    problems = preflight.find_problems(
        [], required=("definitely_not_installed",),
    )
    assert len(problems) == 1
    assert "definitely_not_installed" in problems[0]


def test_preflight_flags_missing_inputs(tmp_path):
    problems = preflight.find_problems([str(tmp_path / "nope")], required=())
    assert problems and "missing input" in problems[0]


def test_preflight_is_quiet_when_everything_is_present(tmp_path):
    assert preflight.find_problems([str(tmp_path)], required=("os",)) == []


def test_missing_unsloth_is_a_warning_not_a_blocker(tmp_path):
    """The whole point of the backend fallback: no unsloth must not stop a run."""
    absent = lambda name: None
    assert preflight.find_problems([str(tmp_path)], find_spec=absent, required=()) == []
    warnings = preflight.find_warnings(find_spec=absent)
    assert any("transformers backend" in w for w in warnings)


def test_unsloth_on_the_wrong_python_is_warned_about():
    present = lambda name: object()
    warnings = preflight.find_warnings(find_spec=present, python=(3, 12))
    assert any("not 3.11" in w for w in warnings)


def test_unsloth_on_the_right_python_is_quiet():
    present = lambda name: object()
    assert preflight.find_warnings(find_spec=present, python=(3, 11)) == []


def test_preflight_survives_a_module_that_raises_on_lookup():
    def exploding_find_spec(name):
        raise ValueError("namespace package without __path__")

    problems = preflight.find_problems(
        [], required=("weird",), find_spec=exploding_find_spec,
    )
    assert len(problems) == 1


def test_decoder_survives_a_truncated_result_file(tmp_path):
    store = tmp_path / "outputs"
    store.mkdir()
    with bz2.BZ2File(store / "aaa_0.p1", "w") as f:
        pickle.dump([guess([[1]])], f)
    (store / "bbb_0.p1").write_bytes(b"not a bz2 file")

    decoder = ArcDecoder(FakeDataset({}, {}), n_guesses=2)
    decoder.load_decoded_results(str(store))
    assert "aaa_0" in decoder.decoded_results
    assert "bbb_0" not in decoder.decoded_results


def plan_steps(task_cap, ttt_fraction, step_seconds, n_samples, epochs):
    """Mirrors the sizing in arc_solver.worker, which cannot be imported here."""
    affordable = int(task_cap * ttt_fraction / step_seconds)
    return int(np.clip(affordable, 8, n_samples * epochs))


def test_a_slow_task_gets_a_shorter_but_complete_schedule():
    assert plan_steps(500, 0.6, 4.0, 128, 1) == 75


def test_a_fast_task_still_caps_at_the_dataset():
    assert plan_steps(500, 0.6, 0.1, 128, 1) == 128
    assert plan_steps(500, 0.6, 0.1, 128, 2) == 256


def test_an_extremely_slow_task_keeps_a_floor():
    assert plan_steps(300, 0.6, 100.0, 128, 1) == 8


def plan_from_tokens(task_cap, ttt_fraction, task_tokens, tokens_per_second,
                     n_samples, epochs):
    """Mirrors the sizing in arc_solver.worker, which cannot be imported here."""
    step_seconds = task_tokens / tokens_per_second
    affordable = int(task_cap * ttt_fraction / step_seconds)
    return int(np.clip(affordable, 8, n_samples * epochs))


def test_step_time_is_predicted_from_prompt_length():
    """Across dev tasks, steps/second times prompt length held near constant at
    ~950 tokens/s whether the prompt was 531 tokens or 6840. Averaging step
    times across those two would be wrong for both."""
    for tokens, observed_steps_per_second in [
        (531, 1.56), (1978, 0.51), (2070, 0.50),
        (3176, 0.32), (3849, 0.24), (6840, 0.13),
    ]:
        predicted = 950.0 / tokens
        assert 0.6 < predicted / observed_steps_per_second < 1.5


def test_a_long_task_is_given_fewer_steps_than_a_short_one():
    short = plan_from_tokens(372, 0.6, 531, 950.0, 128, 1)
    long = plan_from_tokens(372, 0.6, 6840, 950.0, 128, 1)
    assert short > long
    assert short == 128
    assert long == 30


def test_the_plan_fits_the_budget_it_was_given():
    """The point of planning is that the schedule finishes: planned steps times
    predicted step time must not exceed the training share of the budget."""
    for tokens in (531, 1978, 3849, 6840):
        planned = plan_from_tokens(372, 0.6, tokens, 950.0, 128, 1)
        predicted_runtime = planned * tokens / 950.0
        assert predicted_runtime <= 372 * 0.6 + 1e-6 or planned == 128


def test_the_floor_still_applies_to_an_enormous_prompt():
    assert plan_from_tokens(372, 0.6, 100000, 950.0, 128, 1) == 8
