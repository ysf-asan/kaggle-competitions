import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp

import arc_config
from arc_confidence import select_pass2_queue


def local_worker(rank, queue, end_time, pass_id, n_workers):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}_p{pass_id}"):
            time.sleep(5)

    try:
        import unsloth
    except ImportError:
        pass

    from arc_solver import worker

    with open(f"/kaggle/worker{rank}_p{pass_id}", "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start pass {pass_id}!")

    worker(rank, queue, end_time, pass_id=pass_id, n_workers=n_workers)

    print(f"[Rank {rank}] done pass {pass_id}!")


def run_pass(pass_id, task_ids, end_time, n_workers):
    queue = mp.Manager().Queue()
    for key in task_ids:
        queue.put(key)
    for _ in range(n_workers):
        queue.put(None)

    print(f"*** Pass {pass_id}: {len(task_ids)} tasks, {n_workers} workers, "
          f"{(end_time - time.time())/3600:.2f}h budget.", flush=True)

    mp.spawn(local_worker, args=(queue, end_time, pass_id, n_workers),
             nprocs=n_workers)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    task_ids = sorted(data.keys())
    if not rerun_mode:
        task_ids = [k for k in task_ids if k in arc_config.DEV_TASK_IDS]

    n_workers = arc_config.resolve_num_workers(torch.cuda.device_count())
    print(f"*** {torch.cuda.device_count()} GPU(s) visible, using {n_workers} worker(s).")

    global_end = args.end_time
    if not rerun_mode:
        global_end = min(global_end,
                         time.time() + arc_config.DEV_TIME_BUDGET_SECONDS)
        print(f"*** dev run: {arc_config.DEV_TIME_BUDGET_SECONDS/60:.0f} min budget "
              f"(the rerun gets the full 12h)")

    total_budget = global_end - time.time()
    pass1_end = time.time() + total_budget * arc_config.PASS1_TIME_FRACTION

    run_pass(1, task_ids, pass1_end, n_workers)

    if len(arc_config.PASS_BUDGETS) > 1 and time.time() < global_end:
        expected = 8 * arc_config.PASS_BUDGETS[1].eval_permutations
        pass2_ids = select_pass2_queue(
            task_ids,
            arc_config.OUTPUT_DIRS[1],
            expected_results=expected,
            fraction=arc_config.PASS2_TASK_FRACTION,
        )
        if pass2_ids:
            run_pass(2, pass2_ids, global_end, n_workers)
        else:
            print("*** Pass 2: nothing to revisit.")
    else:
        print("*** Pass 2 skipped (baseline mode or out of time).")
