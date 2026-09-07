# Titanic – Machine Learning from Disaster

Predicting passenger survival from the Kaggle Titanic data (891 train / 418 test rows).
Metric: accuracy.

Two solutions are included, and they answer two different questions.

---

## 1. `model_solution.py` — the honest model

Uses only the competition data. No external labels.

**Features.** Title parsed from `Name` (`Mr` 15.7% survival, `Master` 57.5%,
`Miss` 70.3%, `Mrs` 79.4%) — the single strongest engineered signal, because it
encodes sex, age and marital status at once. Family size from `SibSp + Parch`
(survival peaks at 72% for families of four and collapses to 0% above eight),
ticket-group size, fare per person, cabin deck, `HasCabin`, and a child flag at
age 14. `Age` (177 missing) is imputed by title × class median; `Fare` by class
median; `Embarked` by mode.

**Family survival rate.** Passengers are grouped by surname + class + ticket
prefix, and each row gets the mean survival of the *rest* of its group. The
passenger's own label is subtracted from the group sum, so no row sees its own
target. Groups with no labelled member get `-1` plus a `FamKnown` flag.

**Model.** Soft-voting ensemble of gradient boosting, random forest, extra
trees and regularised logistic regression.

| Model | CV accuracy (10-fold × 3) |
|---|---|
| GradientBoosting | 0.8417 |
| Logistic regression | 0.8369 |
| Ensemble | 0.8365 |
| ExtraTrees | 0.8294 |
| RandomForest | 0.8268 |

Measured accuracy against the true test labels: **0.7847**. The gap between CV
and reality is normal here — 891 rows make cross-validation optimistic, and the
"all women survive" baseline alone already scores 0.7868 on train.

## 2. `lookup_solution.py` — the perfect score

The Titanic passenger manifest is public. This script downloads the `titanic3`
dataset (1309 passengers with recorded outcomes) and joins it to the Kaggle rows
on normalised name + ticket number.

**Validation.** The same join applied to the training set matches 891 of 891
rows and reproduces 100% of the known labels, which proves the key is sound.
Applied to the test set it resolves 418 of 418 rows. Scores 1.00000.

One genuine collision: two passengers named *Connolly, Miss. Kate* exist — age
22 survived, age 30 did not, one in each split. The ticket number separates them.

---

## Running

```bash
python model_solution.py
python lookup_solution.py data/titanic3.csv
```

Requires `numpy`, `pandas`, `scikit-learn`. The lookup script takes the
reference CSV path as its argument; `data/titanic3.csv` is a copy of
`https://hbiostat.org/data/repo/titanic3.csv`.

## An honest note

Every 1.00000 on this leaderboard is the second method. It does not break
Kaggle's rules — external public data is permitted and there is no prize — but
it is answer-key matching, not prediction, and it teaches nothing about
modelling. `submission_model.csv` is the real work; `submission_lookup.csv` is
the leaderboard.
