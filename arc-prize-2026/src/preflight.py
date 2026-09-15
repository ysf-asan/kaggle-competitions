"""Fail fast on what actually blocks a run, and only warn about the rest.

The solver imports its model backend inside spawned worker processes, so a
missing input surfaces as a ProcessRaisedException several minutes in with the
real cause buried in a spawn traceback. These checks run in seconds, before
anything is spawned.

unsloth is deliberately *not* required. It gives a faster path when present;
when it is absent the solver runs on plain transformers + peft.
"""

import importlib.util
import os
import sys

UNSLOTH_PYTHON = (3, 11)
DOCKER_IMAGE_VERSION_ID = 31090


def _importable(name, find_spec=None):
    find_spec = find_spec or importlib.util.find_spec
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def describe_inputs(root="/kaggle/input"):
    """What is actually mounted, so a missing input is obvious at a glance."""
    if not os.path.isdir(root):
        return [f"{root} does not exist — no inputs are attached at all"]
    entries = sorted(os.listdir(root))
    if not entries:
        return [f"{root} is empty — no inputs are attached"]
    return [f"{root}/{e}" for e in entries]


def find_problems(required_paths, find_spec=None, required=("torch", "transformers", "peft", "datasets")):
    """Blocking problems only. Empty list means the run can start."""
    problems = []

    for name in required:
        if not _importable(name, find_spec):
            problems.append(f"'{name}' is not importable; it should be in the Kaggle image.")

    for path in required_paths:
        if not os.path.exists(path):
            problems.append(
                f"missing input: {path}\n"
                f"      currently attached: " + ", ".join(describe_inputs()) + "\n"
                f"      add it under Add Input in the notebook editor "
                f"(models may need a licence accepted first)."
            )

    return problems


def find_model_problem(config):
    """Resolve the checkpoint, or explain why we cannot."""
    path = config.resolve_model_path()
    if path is not None:
        return None, path
    return (
        "no model checkpoint found under /kaggle/input.\n"
        "      expected: " + config.MODEL_PATH + "\n"
        "      currently attached: " + ", ".join(describe_inputs()) + "\n"
        "      add the model '" + config.MODEL_OWNER + "/" + config.MODEL_SLUG + "' "
        "(transformers / bfloat16) under Add Input."
    ), None


def find_warnings(find_spec=None, python=None):
    """Things worth knowing that do not stop the run."""
    python = python or sys.version_info[:2]
    warnings = []

    has_unsloth = _importable("unsloth", find_spec)

    if not has_unsloth:
        warnings.append(
            "unsloth not found — using the transformers backend. Same method, "
            "slower per task. To get the fast path, fork a notebook pinned to "
            f"Kaggle image {DOCKER_IMAGE_VERSION_ID} with the "
            "sorokin/pip-install-unsloth-flash-patch input attached."
        )
    elif python != UNSLOTH_PYTHON:
        want = ".".join(map(str, UNSLOTH_PYTHON))
        got = ".".join(map(str, python))
        warnings.append(
            f"unsloth is present but this is Python {got}, not {want}. If it "
            f"fails to import, the solver falls back to transformers."
        )

    return warnings


def check(required_paths):
    import arc_backend
    import arc_config

    print(f"python {sys.version.split()[0]}")
    for name in ("torch", "transformers", "peft", "datasets", "unsloth", "flash_attn"):
        print(f"  {name:12s} {'ok' if _importable(name) else 'missing'}")

    try:
        import torch
        print(f"  cuda devices {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  cuda devices unknown ({e})")

    print("attached inputs:")
    for line in describe_inputs():
        print(f"  {line}")

    print(f"*** backend: {arc_backend.name()}")

    for w in find_warnings():
        print(f"*** note: {w}")

    problems = find_problems(required_paths)

    model_problem, model_path = find_model_problem(arc_config)
    if model_problem:
        problems.append(model_problem)
    else:
        print(f"*** model: {model_path}")
        if model_path != arc_config.MODEL_PATH:
            print(f"*** note: resolved by search; MODEL_PATH says {arc_config.MODEL_PATH}")

    if problems:
        raise RuntimeError(
            "Cannot start:\n  - " + "\n  - ".join(problems)
        )
    print("*** preflight ok")
