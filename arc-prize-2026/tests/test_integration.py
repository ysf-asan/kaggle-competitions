"""Integration checks that need the real ArcDataset but no model.

Two things are worth pinning down before spending a submission on them: that
`build_decode_batches` slices the keys `ArcDataset.augment` actually produces
(the batching indexes groups by position, so a change in key naming would
silently scramble it), and that a partially finished run still writes a
submission Kaggle will accept.
"""

import os
import sys
import bz2
import json
import pickle
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arc_loader import ArcDataset
from arc_batching import build_decode_batches
from arc_decoder import ArcDecoder

DATA = os.path.join(os.path.dirname(__file__), "..",
                    "arc-prize-2026-arc-agi-2")
EVAL_CHALLENGES = os.path.join(DATA, "arc-agi_evaluation_challenges.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(EVAL_CHALLENGES),
    reason="competition data not present",
)


def load_eval(keys=None):
    return ArcDataset.from_file(EVAL_CHALLENGES, keys=keys)


def augmented_subkeys(task_id, n_perm):
    ds = load_eval().change_keys([task_id]).split_multi_replies()
    aug = ds.augment(n=n_perm, seed=2)
    by_test = {}
    for subkey in sorted(aug.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        by_test.setdefault(test_id, []).append(subkey)
    return by_test


@pytest.mark.parametrize("n_perm", [2, 4])
def test_augment_lays_out_eight_contiguous_dihedral_groups(n_perm):
    by_test = augmented_subkeys("0934a4d8", n_perm)
    subkeys = by_test["0"]

    assert len(subkeys) == 8 * n_perm

    def transform_of(subkey):
        return tuple(op for op in subkey.split(".")[1:]
                     if not op.startswith("permute") and not op.startswith("ex"))

    for g in range(8):
        group = subkeys[g * n_perm:(g + 1) * n_perm]
        transforms = {transform_of(k) for k in group}
        assert len(transforms) == 1, (
            f"group {g} is not contiguous under sorting: {transforms}")

    groups = [transform_of(subkeys[g * n_perm]) for g in range(8)]
    assert len(set(groups)) == 8, "the eight dihedral views must be distinct"


@pytest.mark.parametrize("n_perm", [2, 4])
def test_batches_over_real_keys_span_two_groups_and_cover_everything(n_perm):
    by_test = augmented_subkeys("0934a4d8", n_perm)
    batches = build_decode_batches(by_test, n_perm)

    flat = [k for b in batches for k in b]
    assert sorted(flat) == sorted(by_test["0"])
    assert all(len(b) == 4 for b in batches)

    def transform_of(subkey):
        return tuple(op for op in subkey.split(".")[1:]
                     if not op.startswith("permute") and not op.startswith("ex"))

    for batch in batches:
        assert len({transform_of(k) for k in batch}) == 2


@pytest.mark.parametrize("n_perm", [2, 4])
def test_truncation_does_not_scramble_the_group_layout(n_perm):
    """`cut_to_len` rewrites the trailing `.ex...` segment of the keys it
    shortens. The batching indexes groups by position in the sorted key list,
    so that rewrite must not move a key out of its dihedral group."""
    ds = load_eval().change_keys(["981571dc"]).split_multi_replies()
    aug = ds.augment(n=n_perm, seed=2)

    key0 = aug.keys[0]
    full = aug.get_length(key0, formatter=None, name="input")
    n_pairs = len(aug.queries[key0]["train"])
    floor_len = full - sum(
        np.prod(np.shape(v)) for p in aug.queries[key0]["train"] for v in p.values()
    )
    max_len = int(floor_len + (full - floor_len) / max(2, n_pairs))
    assert floor_len < max_len < full

    cut = aug.cut_to_len(formatter=None, name="input", max_len=max_len)

    subkeys = sorted(cut.keys)
    assert len(subkeys) == 8 * n_perm
    assert any(k.split(".")[-1] != a.split(".")[-1]
               for k, a in zip(subkeys, sorted(aug.keys))), "nothing was truncated"

    def transform_of(subkey):
        return tuple(op for op in subkey.split(".")[1:]
                     if not op.startswith("permute") and not op.startswith("ex"))

    for g in range(8):
        group = subkeys[g * n_perm:(g + 1) * n_perm]
        assert len({transform_of(k) for k in group}) == 1, (
            f"group {g} lost contiguity after truncation")


def prompt_token_estimate(query):
    """Digits plus one newline per row, over every grid in the prompt.

    The formatter writes one character per cell and a newline per row, so this
    tracks the tokenised prompt length exactly up to fixed per-pair overhead.
    """
    total = 0
    for pair in query["train"]:
        for grid in (pair["input"], pair["output"]):
            total += len(grid) * len(grid[0]) + len(grid)
    test_in = query["test"][0]["input"]
    total += len(test_in) * len(test_in[0]) + len(test_in)
    return total


@pytest.mark.parametrize("task_id", ["0934a4d8", "981571dc", "20270e3b"])
@pytest.mark.parametrize("n_perm", [2, 4])
def test_every_batch_has_uniform_prompt_length(task_id, n_perm):
    """The invariant that keeps decoding from crashing.

    inference_turbo_dfs stacks the batch's prefixes into one tensor, which is
    only possible if they tokenise to the same length. Rotating by 90 degrees
    swaps a grid's rows and columns and therefore changes its newline count, so
    a batch may only mix transforms that preserve the shape. That is exactly
    why groups are paired two apart — a and a+2 differ by a 180 degree turn.
    """
    ds = load_eval().change_keys([task_id]).split_multi_replies()
    aug = ds.augment(n=n_perm, seed=2)

    by_test = {}
    for subkey in sorted(aug.keys):
        by_test.setdefault(subkey.split(".")[0].split("_")[1], []).append(subkey)

    batches = build_decode_batches(by_test, n_perm)
    assert batches

    for batch in batches:
        lengths = {prompt_token_estimate(aug.queries[k]) for k in batch}
        assert len(lengths) == 1, (
            f"batch mixes prompt lengths {sorted(lengths)}; the decoder cannot "
            f"stack these into one tensor"
        )


def test_multi_output_task_is_batched_per_test_input():
    by_test = augmented_subkeys("13e47133", 2)
    assert len(by_test) > 1, "expected a task with more than one test input"
    batches = build_decode_batches(by_test, 2)
    firsts = [b[0].split(".")[0] for b in batches[:len(by_test)]]
    assert len(set(firsts)) == len(by_test)


def test_partial_run_still_produces_a_complete_submission(tmp_path):
    """A run cut short must still name every task id with both attempts."""
    task_ids = sorted(json.load(open(EVAL_CHALLENGES)).keys())[:6]
    data = load_eval(keys=task_ids)
    multi = data.split_multi_replies()

    solved = multi.keys[0]
    store = tmp_path / "outputs"
    store.mkdir()
    grid = np.array([[1, 2], [3, 4]])
    with bz2.BZ2File(store / f"{solved}.permute0123456789", "w") as f:
        pickle.dump([{"solution": grid, "beam_score": 0.5,
                      "score_aug": [1.0] * 8}], f)

    decoder = ArcDecoder(multi, n_guesses=2)
    decoder.load_decoded_results(str(store))
    submission = data.get_submission(decoder.run_selection_algo())

    assert set(submission.keys()) == set(task_ids)
    for k in task_ids:
        assert len(submission[k]) == len(data.queries[k]["test"])
        for output in submission[k]:
            assert "attempt_1" in output and "attempt_2" in output

    solved_task, solved_index = solved.rsplit("_", 1)
    assert submission[solved_task][int(solved_index)]["attempt_1"] == grid.tolist()

    json.loads(json.dumps(submission))


def grid_token_count(grid):
    """One token per cell plus one newline per row, as the formatter writes."""
    return len(grid) * len(grid[0]) + len(grid)


def synthetic_transcript(pairs, header=(14, 10), end=15):
    """The id stream fmt_train produces, with placeholder content tokens."""
    ids = []
    for p in pairs:
        ids += list(header) + [1] * grid_token_count(p["input"]) + [end]
        ids += list(header) + [2] * grid_token_count(p["output"]) + [end]
    return ids


@pytest.mark.parametrize("task_id", [
    "20270e3b", "0934a4d8", "13e47133", "36a08778",
    "aa4ec2a5", "db0c5428", "981571dc", "d8e07eb2",
])
def test_supervision_matches_the_output_grids_of_real_tasks(task_id):
    """Ties the label masking to the data rather than to a synthetic example.

    The supervised count should equal the size of the demonstration outputs,
    plus one end marker each — nothing more. A task with huge inputs and tiny
    outputs should come out at a few percent, and that is correct rather than a
    symptom of the masking having gone wrong again.
    """
    from arc_labels import build_completion_labels, IGNORE_INDEX

    pairs = load_eval().queries[task_id]["train"]
    ids = synthetic_transcript(pairs)
    labels = build_completion_labels(ids, [14, 10], [15], user_header=[14, 10])

    supervised = sum(1 for l in labels if l != IGNORE_INDEX)
    expected = sum(grid_token_count(p["output"]) + 1 for p in pairs)
    assert supervised == expected

    assert all(labels[i] in (2, 15)
               for i, l in enumerate(labels) if l != IGNORE_INDEX)
