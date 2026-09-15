"""Confidence scoring of pass-1 results, used to pick the pass-2 queue.

Kept free of torch/unsloth imports so it can be unit tested off-GPU.
"""

import os
import bz2
import pickle
import numpy as np


def hashable(guess):
    return tuple(map(tuple, guess))


def _group_by_solution(guesses):
    groups = {}
    for g in guesses:
        groups.setdefault(hashable(g["solution"]), []).append(g)
    return groups


def getter_kgmon(guesses):
    """Baseline ranking score: how many decode results produced this exact grid,
    minus the mean augmentation NLL of those results (lower NLL is better)."""
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def output_confidence(guesses, expected_results):
    """Confidence in [0, 1] for one test output.

    Three signals, all cheap and already present in the decode results:

    - coverage: did the decoder produce anything at all? A task where the DFS
      returned two beams out of sixteen attempts is not "confident", it is
      starved, and starved tasks are exactly what pass 2 should revisit.
    - support: what share of the decode results agree on the top grid.
    - margin: how far the top grid outscores the runner-up. Undefined when
      everything agreed, which is the good case, so it is treated as maximal.
    """
    if not guesses:
        return 0.0

    groups = _group_by_solution(guesses)
    scored = sorted((getter_kgmon(g) for g in groups.values()), reverse=True)

    n_results = len(guesses)
    coverage = min(1.0, n_results / max(1, expected_results))
    support = max(len(g) for g in groups.values()) / n_results

    if len(scored) > 1:
        margin = float(np.clip((scored[0] - scored[1]) / 4.0, 0.0, 1.0))
    else:
        margin = 1.0

    return coverage * (0.6 * support + 0.4 * margin)


def load_pass_results(store):
    """subkey-store -> {base_key: [guess, ...]}, base_key being `{task}_{index}`."""
    results = {}
    if not os.path.isdir(store):
        return results
    for name in os.listdir(store):
        try:
            with bz2.BZ2File(os.path.join(store, name)) as f:
                outputs = pickle.load(f)
        except Exception as e:
            print(f"*** Skipping unreadable result '{name}': {e}")
            continue
        results.setdefault(name.split(".")[0], []).extend(outputs)
    return results


def rank_tasks(task_ids, store, expected_results):
    """Rank every task from least to most confident.

    Tasks pass 1 never reached have no results and sort first, ahead of every
    task that produced something. A task with several test outputs is only as
    confident as its weakest output.
    """
    per_output = load_pass_results(store)

    by_task = {}
    for base_key, guesses in per_output.items():
        task_id = base_key.rsplit("_", 1)[0]
        conf = output_confidence(guesses, expected_results)
        by_task[task_id] = min(conf, by_task.get(task_id, 1.0))

    ranked = [(by_task.get(t, -1.0), t) for t in task_ids]
    ranked.sort(key=lambda x: (x[0], x[1]))
    return ranked


def select_pass2_queue(task_ids, store, expected_results, fraction):
    """Least-confident `fraction` of the tasks, plus every unreached task."""
    ranked = rank_tasks(task_ids, store, expected_results)

    unreached = [t for conf, t in ranked if conf < 0.0]
    reached = [t for conf, t in ranked if conf >= 0.0]

    n_take = int(round(fraction * len(reached)))
    return unreached + reached[:n_take]
