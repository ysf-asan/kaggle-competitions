import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
import arc_config

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

if rerun_mode:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

for pass_id, store in sorted(arc_config.OUTPUT_DIRS.items()):
    decoder.load_decoded_results(store, run_name=arc_config.RUN_NAMES[pass_id])

n_results = sum(len(v) for v in decoder.decoded_results.values())
print(f"*** {n_results} decode results over {len(decoder.decoded_results)} outputs")
if rerun_mode and n_results == 0:
    raise RuntimeError(
        "no decode results at all: every prediction would be a placeholder and "
        "the submission would score zero. Check the solver output above."
    )

submission = data.get_submission(decoder.run_selection_algo())

with open("submission.json", "w") as f:
    json.dump(submission, f)

assert set(submission.keys()) == set(data.keys), "submission is missing task ids"
for k, outputs in submission.items():
    assert len(outputs) == len(data.queries[k]["test"]), f"{k}: wrong number of outputs"
    for o in outputs:
        assert "attempt_1" in o and "attempt_2" in o, f"{k}: missing attempt"
print(f"*** submission.json written for {len(submission)} tasks.")

if not rerun_mode:
    decoder.benchmark_selection_algos()
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score (contaminated, dev signal only):", data.validate_submission(reload_submission))
