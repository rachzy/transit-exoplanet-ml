# transit-exoplanet-ml

A reproducible screening model for transit exoplanet candidates. It validates
processed per-star candidate tables, evaluates a star-grouped stacked
classifier with nested cross-validation, trains the production stack, and
writes one consolidated prediction CSV.

> **`potential_probability` is a screening score learned from
> literature-derived labels. It is not scientific confirmation.** A `POTENTIAL`
> flag says the candidate resembles candidates that were later confirmed in the
> training literature; it says nothing about whether the planet exists.

## Install

The project uses [uv](https://docs.astral.sh/uv/), `pyproject.toml`, and a
committed `uv.lock`. It targets CPython 3.14 and supports 3.11 – 3.14.

```bash
uv sync
uv run exoplanet-ml --help
```

Installation and the test suite do not depend on `data/` being populated.

## Data contract

Each file holds the candidates for exactly one star and is named
`<star-name>_<YYYYMMDD>.csv`. Everything before the **final** underscore is the
`star_id`, so star names may contain underscores themselves.

| Rule | Behaviour |
| --- | --- |
| Malformed filename | rejected |
| Two files for one star in a dataset | rejected |
| Unknown column | rejected until the schema is explicitly updated |
| Missing feature column | rejected |
| `candidate_label` / `detection_status` | required for `train`, optional for `predict` |
| Invalid label or status value | rejected |
| Duplicate candidate rows in a file | rejected |
| Infinite values | rejected |
| Feature empty across the whole dataset | rejected |
| Missing values (`NaN`) | permitted, imputed inside model folds |

`candidate_label` maps `CONFIRMED → 1` and `FALSE-POSITIVE → 0`. Rows are
weighted so **every star contributes the same total weight to any fit it takes
part in**; the weights are recomputed per fitting subset, so the property also
holds for the accepted-only meta-model fit.

### Feature allowlist

The schema in [`src/resources/schema.yaml`](src/resources/schema.yaml)
is versioned and strict: exactly 36 columns reach a model. Everything else in
the CSVs is withheld, for a stated reason:

| Group | Columns | Why |
| --- | --- | --- |
| supervision / metadata | `candidate_label`, `detection_status`, `matched_target`, `matched_period_ratio` | they encode the answer |
| pipeline control | `mes_threshold_used`, `is_provisional_detection` | settings of the detection run, not properties of the candidate |
| non-generalizable epoch | `t0` | an absolute time with no transferable meaning |
| exact aliases | `duration_days`, `scale_skewness`, `scale_kurtosis`, `scale_outlier_resistance`, `snr_per_transit_mean`, `snr_per_transit_std`, `planet_radius_rjup` | duplicates (up to a unit conversion) of retained features |

Changing `feature_columns` requires bumping `schema_version`; model bundles
refuse to load against a schema they were not trained on.

## Commands

```bash
# 1. Check a directory against the contract
uv run exoplanet-ml validate --mode train   --data-dir data/processed/train
uv run exoplanet-ml validate --mode predict --data-dir data/processed/test

# 2. Nested, star-grouped evaluation
uv run exoplanet-ml evaluate --data-dir data/processed/train --output-dir reports/nested-cv

# 3. Train the production stack into an immutable run directory
uv run exoplanet-ml train --data-dir data/processed/train --artifact-dir models

# 4. Score unseen stars
uv run exoplanet-ml predict \
    --model-dir models/<run-id> \
    --data-dir data/processed/test \
    --output predictions.csv
```

`train` runs the nested evaluation first so the saved artifact carries honest
metrics; pass `--skip-evaluation` to skip that during iteration. Both `evaluate`
and `train` accept `--config` and `--schema` to override the packaged YAML.

`evaluate --output-dir` writes `metrics.json`, `model_comparison.csv`,
`oof_predictions.csv`, `per_star.csv`, `permutation_importance.csv` and its
summary, `provenance.json`, and a `plots/` directory. `train` nests the same set
under `evaluation/` inside the run directory.

### Prediction output

The CSV keeps every original candidate column and appends:

`source_file`, `star_id`, `row_index`, `prob_lightgbm`, `prob_extra_trees`,
`prob_svm_rbf`, `prob_logistic_regression`, `potential_probability`,
`decision_threshold`, `prediction` (`POTENTIAL` / `UNLIKELY`), `model_run_id`.

Rows are ordered by source file then by their position within that file, so the
output is byte-stable across runs.

## Python API

```python
from src import (
    load_dataset, validate_dataset, evaluate_dataset,
    train_model, load_model, predict_dataset,
)

dataset = load_dataset("data/processed/train", mode="train")
result  = evaluate_dataset(dataset=dataset)
run     = train_model(dataset=dataset)
frame   = predict_dataset(run.bundle, data_dir="data/processed/test")
```

## Modeling

**The production model is always the LightGBM + ExtraTrees + RBF SVM +
logistic-regression stack with an L2 logistic meta-model**, even when a simpler
comparator scores higher. The `evaluate` and `train` commands print a warning
naming any comparator that beats the stack, and the artifact records it.

Each base learner gets its own preprocessing, all of it fitted inside the fold
that uses it and persisted with the bundle:

| Learner | Preprocessing |
| --- | --- |
| LightGBM | none — native missing-value handling |
| ExtraTrees | drop fold-local degenerates → median imputation + missing indicators |
| RBF SVM, logistic regression | the above, then robust scaling |

Base learners train on **all** candidates, accepted and rejected. The
meta-model is fitted **only** on accepted candidates' out-of-fold base
probabilities, and only the four probabilities reach it. No automatic class
weighting is used beyond the equal-star weights; the high-recall objective is
met by model selection and thresholding instead.

### Objective

Pick the threshold with the greatest precision that still reaches **95 % recall**
on cross-fitted **accepted** candidates. Ties resolve towards higher recall and
then towards the lower (more inclusive) threshold.

Reported metrics and the tuning score use the same equal-star weights as the
fits, so a star with many candidates does not dominate the numbers
(`objective.weighted_metrics`, on by default).

### Nested evaluation

Splits are `StratifiedGroupKFold`, grouped by `star_id` and stratified on the
composite of label and acceptance status: five outer folds for honest
reporting, three inner folds for tuning and out-of-fold base probabilities.
Fold counts are reduced only when class/star support demands it, never below
three outer or two inner folds — otherwise the run fails with a diagnostic
naming the thin strata.

The meta-model is cross-fitted as well as the bases, so each evaluation fold
picks its threshold from its own outer-training predictions only, then applies
it to untouched outer-validation stars. Permutation importance is measured
solely on held-out stars and aggregated across folds.

Tuning is conservative and seeded (`seed: 42`), scored by accepted-candidate
average precision: 6 logistic-regression candidates, 24 each for the RBF SVM,
ExtraTrees (500 trees), and LightGBM. The meta-model is fixed at L2 with
`C = 0.1` and is never tuned.

## Artifacts

Every training run writes an immutable `models/<run-id>/` directory:

```
model.joblib                    serialized stack + every fitted pipeline
config.resolved.yaml            the configuration exactly as applied
feature_schema.yaml             the feature schema, with its version
threshold.json                  saved threshold and each comparator's
training_report.json            cross-fitted metrics and the recall-floor check
tuning.json                     every candidate scored, per learner
training_stars.json             training stars + input file checksums
training_oof_predictions.csv    cross-fitted training predictions
per_star.csv                    per-star breakdown
provenance.json                 run id, seed, git commit, dependency versions
summary.json                    headline numbers
MANIFEST.json                   sha256 of every file above
evaluation/                     nested-CV metrics, OOF predictions,
                                permutation importance, and plots
```

Bundles are Joblib pickles: load only bundles produced by your own pipeline,
from a location you trust.

## Development

```bash
uv run pytest              # 173 tests, no repository data required
uv run ruff check src
uv run coverage run -m pytest && uv run coverage report
```

A `Makefile` wraps the common commands above; run `make help` for the list.

Processed CSVs under `data/` are tracked in Git. Virtual environments, caches,
generated models, reports, and predictions are ignored.

## License

See [LICENSE](LICENSE).
