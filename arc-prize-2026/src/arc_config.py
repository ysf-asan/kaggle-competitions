"""Compute budget configuration for the two-pass solver.

Pass 1 runs every task on a cheap budget. The confidence of each pass-1 result
decides which tasks are re-run in pass 2 with a richer budget. Setting
BASELINE=True reproduces the public 33.89 notebook exactly (single pass, the
original hyper-parameters) and is kept as a safety net.
"""

import numpy as np

BASELINE = False

PASS1_TIME_FRACTION = 0.55

PASS2_TASK_FRACTION = 0.40

import os

MODEL_OWNER = "sorokin"
MODEL_SLUG = "qwen3_4b_grids15_sft139"

MODEL_PATH = f"/kaggle/input/models/{MODEL_OWNER}/{MODEL_SLUG}/transformers/bfloat16/1"

INPUT_ROOT = "/kaggle/input"


def find_model_dirs(root=INPUT_ROOT, max_depth=5):
    """Directories under `root` that look like a loadable checkpoint."""
    hits = []
    if not os.path.isdir(root):
        return hits
    for dirpath, dirnames, filenames in os.walk(root):
        depth = os.path.relpath(dirpath, root).count(os.sep)
        if os.path.relpath(dirpath, root) == ".":
            depth = 0
        if depth >= max_depth:
            dirnames[:] = []
        if "config.json" in filenames and any(
            f.endswith((".safetensors", ".bin")) for f in filenames
        ):
            hits.append(dirpath)
            dirnames[:] = []
    return sorted(hits)


def resolve_model_path(root=INPUT_ROOT, expected=None):
    """The checkpoint to load, or None if nothing usable is attached.

    Falls back to searching because the version number in MODEL_PATH is the
    part most likely to drift, and a wrong path otherwise surfaces deep inside
    transformers as an unhelpful "Repo id must be in the form ..." error: it
    treats a non-existent directory as a Hugging Face repo name and tries to
    download it.
    """
    expected = expected or MODEL_PATH
    if os.path.isdir(expected):
        return expected

    hits = find_model_dirs(root)
    named = [h for h in hits if MODEL_SLUG in h]
    if named:
        return named[0]
    if len(hits) == 1:
        return hits[0]
    return None

NUM_WORKERS = None


def resolve_num_workers(device_count):
    if NUM_WORKERS is not None:
        return NUM_WORKERS
    return max(1, device_count)

DEV_TASK_IDS = [
    "0934a4d8", "36a08778", "981571dc", "aa4ec2a5",
    "20270e3b",
    "db0c5428",
    "d8e07eb2",
    "13e47133",
]

MAX_SEQ_LENGTH = 8192

LORA_RANK = 256

TTT_TIME_FRACTION = 0.6

TOKENS_PER_SECOND_PRIOR = 950.0

DEV_TIME_BUDGET_SECONDS = 900

GLOBAL_RESERVE_SECONDS = 1800

OUTPUT_DIRS = {
    1: "/kaggle/inference_outputs",
    2: "/kaggle/inference_outputs_p2",
}

RUN_NAMES = {1: "", 2: "_p2"}


class PassBudget:
    """Per-pass knobs. `max_score` is a negative-log-probability cutoff for the
    DFS decoder, so a larger value explores more (and costs more)."""

    def __init__(self, ttt_augmentations, ttt_epochs, eval_permutations,
                 max_score, per_task_cap, dfs_cap, adaptive_cap=True):
        self.ttt_augmentations = ttt_augmentations
        self.ttt_epochs = ttt_epochs
        self.eval_permutations = eval_permutations
        self.max_score = max_score
        self.per_task_cap = per_task_cap
        self.dfs_cap = dfs_cap
        self.adaptive_cap = adaptive_cap


BASELINE_BUDGET = PassBudget(
    ttt_augmentations=16,
    ttt_epochs=1,
    eval_permutations=2,
    max_score=-np.log(0.2),
    per_task_cap=1200,
    dfs_cap=540,
    adaptive_cap=False,
)

PASS_BUDGETS = {
    1: PassBudget(
        ttt_augmentations=16,
        ttt_epochs=1,
        eval_permutations=2,
        max_score=-np.log(0.2),
        per_task_cap=600,
        dfs_cap=300,
    ),
    2: PassBudget(
        ttt_augmentations=24,
        ttt_epochs=2,
        eval_permutations=4,
        max_score=-np.log(0.12),
        per_task_cap=1500,
        dfs_cap=540,
    ),
}

if BASELINE:
    PASS_BUDGETS = {1: BASELINE_BUDGET}
    PASS1_TIME_FRACTION = 1.0
    PASS2_TASK_FRACTION = 0.0
