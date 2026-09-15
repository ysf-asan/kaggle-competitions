import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


def infer_shape_rule(train_pairs):
    """Rule that maps an input shape to the expected output shape, or None."""
    if len(train_pairs) < 2:
        return None

    ins = [np.shape(p["input"]) for p in train_pairs]
    outs = [np.shape(p["output"]) for p in train_pairs]
    if any(len(s) != 2 for s in ins + outs):
        return None

    if all(i == o for i, o in zip(ins, outs)):
        return ("same", None)

    if len(set(outs)) == 1:
        return ("const", outs[0])

    ratios = set()
    for (ih, iw), (oh, ow) in zip(ins, outs):
        if ih and iw and oh % ih == 0 and ow % iw == 0:
            ratios.add((oh // ih, ow // iw))
        else:
            ratios = None
            break
    if ratios and len(ratios) == 1:
        return ("scale", ratios.pop())

    ratios = set()
    for (ih, iw), (oh, ow) in zip(ins, outs):
        if oh and ow and ih % oh == 0 and iw % ow == 0:
            ratios.add((ih // oh, iw // ow))
        else:
            ratios = None
            break
    if ratios and len(ratios) == 1:
        return ("div", ratios.pop())

    return None


def expected_shape(rule, input_shape):
    if rule is None:
        return None
    kind, param = rule
    ih, iw = input_shape
    if kind == "same":
        return (ih, iw)
    if kind == "const":
        return param
    if kind == "scale":
        return (ih * param[0], iw * param[1])
    if kind == "div":
        if param[0] and param[1] and ih % param[0] == 0 and iw % param[1] == 0:
            return (ih // param[0], iw // param[1])
    return None


def apply_shape_prior(ordered_outputs, query):
    """Stable partition of the ranked candidates: conforming grids first."""
    if not ordered_outputs or query is None:
        return ordered_outputs

    rule = infer_shape_rule(query.get("train", []))
    target = expected_shape(rule, np.shape(query["test"][0]["input"]))
    if target is None:
        return ordered_outputs

    conforming = [g for g in ordered_outputs if np.shape(g) == target]
    if not conforming or len(conforming) == len(ordered_outputs):
        return ordered_outputs

    rest = [g for g in ordered_outputs if np.shape(g) != target]
    return conforming + rest


class ArcDecoder:

    def __init__(self, dataset, n_guesses, use_shape_prior=True):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.use_shape_prior = use_shape_prior
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        if not os.path.isdir(store):
            print(f"*** No results at '{store}', skipping.")
            return self
        n_loaded = 0
        for key in os.listdir(store):
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                print(f"*** Skipping unreadable result '{key}': {e}")
                continue
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample
                n_loaded += 1
        print(f"*** Loaded {n_loaded} results from '{store}' for {len(self.decoded_results)} outputs.")
        return self

    def run_selection_algo(self, selection_algorithm=score_kgmon, use_shape_prior=None):
        if use_shape_prior is None:
            use_shape_prior = self.use_shape_prior
        selected = {}
        for bk, v in self.decoded_results.items():
            ordered = selection_algorithm({k: g for k, g in v.items()})
            if use_shape_prior:
                ordered = apply_shape_prior(ordered, self.dataset.queries.get(bk))
            selected[bk] = ordered
        return selected

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            for use_prior in [False, True]:
                name = selection_algorithm.__name__ + (" + shape prior" if use_prior else "")
                selected = self.run_selection_algo(selection_algorithm, use_shape_prior=use_prior)
                correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
                score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
                print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
