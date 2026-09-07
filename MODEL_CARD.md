# Model Card — Transit Exoplanet Candidate Screener

## Model details

|                   |                                                                                  |
| ----------------- | -------------------------------------------------------------------------------- |
| **Name**          | transit-exoplanet-ml candidate screener                                          |
| **Version**       | 0.1.0 (feature schema version 1)                                                 |
| **Type**          | Stacked binary classifier                                                        |
| **Base learners** | LightGBM, ExtraTrees (500 trees), RBF SVM (Platt-scaled), L2 logistic regression |
| **Meta-learner**  | L2 logistic regression, `C` selected by grouped inner CV                         |
| **Meta inputs**   | the four base probabilities, and nothing else                                    |
| **Output**        | `potential_probability` ∈ [0, 1] and a `POTENTIAL` / `UNLIKELY` decision         |
| **Seed**          | 42                                                                               |
| **License**       | see `LICENSE`                                                                    |

All configured candidates are fitted on every run and **the highest-scoring candidate
that meets the recall floor ships**. The full ranking, the runner-up, and the winning
margin are recorded in `selection.json` and printed by the CLI. Every candidate
stays in the bundle, so the choice can be revisited without refitting; setting
`selection.strategy` to a model name pins that model instead.

**The reported score for the winner is optimistically biased.** It is the
maximum of six correlated held-out estimates, chosen on the same numbers it is
then reported against. On a few dozen stars the winning margin is routinely
smaller than the bootstrap intervals, which means the ranking itself is unstable
-- a different sample of stars would plausibly crown a different model. Read the
winner as "one of several statistically indistinguishable options", not as an
established best.

## Intended use

Triage of transit candidates that a detection pipeline has already produced:
ranking and shortlisting candidates for human or follow-up attention, with the
operating point deliberately tuned for high recall.

**`potential_probability` is a screening score learned from literature-derived
labels, not scientific confirmation.** The labels say which candidates were
recorded as confirmed planets or false positives in the literature the training
set was built from. A `POTENTIAL` flag means "this resembles candidates that
were later confirmed"; it is not evidence that a planet exists, and it does not
replace vetting, follow-up observation, or peer review.

### Out of scope

- Confirming or refuting a planet.
- Raw light-curve processing, CNNs on pixel or flux time series, and data downloading — none are part of this project.
- Online or low-latency serving; this is a batch tool.
- Stars or instruments unlike those in the training set (see _Limitations_).

## Training data

Processed per-star candidate tables under `data/processed/train/`, one CSV per
star, named `<star-name>_<YYYYMMDD>.csv`. Each row is one candidate.

Two independent axes are used:

- **`candidate_label`** — the target: `CONFIRMED → 1`, `FALSE-POSITIVE → 0`.
- **`detection_status`** — whether the detection pipeline _accepted_ the
  candidate (`accepted`) or not (`rejected`, `provisional`, `harmonic_duplicate`).

Rejected candidates train the base learners, which is where their signal is
useful. They never train the meta-model and never influence the threshold,
because the screening objective is defined over accepted candidates.

Every star contributes the same total weight to any fit it takes part in, so a
star with twelve candidates does not outvote a star with two.

## Features

Exactly 36 numeric columns reach a model, fixed by the versioned allowlist in
`resources/schema.yaml`. Withheld columns and the reason each is withheld:

- **Supervision / metadata** — `candidate_label`, `detection_status`,
  `matched_target`, `matched_period_ratio`. These encode the answer.
- **Pipeline controls** — `mes_threshold_used`, `is_provisional_detection`.
  Properties of the detection run, not of the candidate; a model that learns
  them learns the pipeline's configuration.
- **Non-generalizable epoch** — `t0`, an absolute reference time that carries no
  transferable meaning.
- **Exact aliases** — `duration_days`, `scale_skewness`, `scale_kurtosis`,
  `scale_outlier_resistance`, `snr_per_transit_mean`, `snr_per_transit_std`,
  `planet_radius_rjup`. Each was verified to be an exact duplicate of a retained
  feature, up to a fixed unit conversion.

Missing values are permitted and imputed inside model folds. Infinities,
unknown columns, invalid labels or statuses, duplicate rows, and features that
are empty across the whole dataset are rejected at load time.

## Objective and operating point

Select the threshold with the greatest precision that still achieves **at least
95 % recall** on cross-fitted accepted candidates. Ties resolve towards higher
recall, then towards the lower (more inclusive) threshold.

No automatic class weighting is applied beyond the equal-star sample weights.
The recall target is met through model selection and thresholding, which keeps
the probabilities interpretable rather than skewed by a class-weight prior.

The saved threshold is derived from cross-fitted predictions on the **training**
stars. Held-out recall on genuinely new stars can differ, and the nested
evaluation reports what it was.

## Evaluation methodology

`StratifiedGroupKFold`, grouped by `star_id` and stratified on the composite of
label and acceptance status:

- **Five outer folds** for honest reporting.
- **Three inner folds** for tuning and out-of-fold base probabilities.
- Fold counts drop only when class/star support requires it, never below three
  outer or two inner folds; otherwise the run fails with a diagnostic naming the
  under-supported strata.

Grouping by star matters: candidates from one star share a light curve, a
detection pipeline run, and a stellar host, so a row-level split would leak.

Both the base learners **and the meta-model** are cross-fitted. Each evaluation
fold selects its threshold using only its outer-training predictions and then
applies it to untouched outer-validation stars. Permutation importance is
computed solely on held-out stars, and the per-fold distributions are aggregated.

Reported for the stack and every configured base learner, plus the
unweighted probability-average baseline: precision and recall at the chosen threshold,
average precision, ROC-AUC, F2, confusion matrix, Brier score, log loss, a
calibration curve, per-star summaries, and star-bootstrap 95 % confidence
intervals.

## Results

Run `20260824T181710Z-256238b4`, seed 42, schema version 1.

### Training set

| | |
| --- | --- |
| Stars | 35 |
| Candidates | 228 |
| Accepted candidates | 87 (71 CONFIRMED / 16 FALSE-POSITIVE) |
| Non-accepted candidates | 141 (base-learner training only) |
| Model features | 36 |

Accepted candidates are **82% CONFIRMED**. That is the precision a "flag everything" rule would achieve at 100 % recall, and it is the number every precision below should be read against.

### Selection

**Shipped model: `lightgbm`**, chosen by average_precision on nested evaluation.

| Rank | Model | average_precision |
| --- | --- | --- |
| 1 | LightGBM **← shipped** | 0.978 |
| 2 | ExtraTrees | 0.962 |
| 3 | Probability average | 0.951 |
| 4 | RBF SVM | 0.915 |
| 5 | Logistic regression | 0.894 |
| 6 | Stack | 0.893 |

The winning margin over the runner-up (ExtraTrees) is **0.0160**, against a 95 % bootstrap interval for the winner of 0.949 – 0.995. The margin is far smaller than that interval, so the ranking is not stable: a different sample of stars would plausibly select a different model. The winner should be read as one of several statistically indistinguishable options, and its headline score as optimistic -- it is the maximum of 6 correlated estimates chosen on the same numbers reported for it.

### Nested evaluation

5 outer folds, inner folds [3, 3, 3, 3, 3], grouped by star. Each fold picked its own threshold from its outer-training predictions only. Intervals are percentile 95 % CIs from 1000 star-level bootstrap resamples.

| Model | Precision | 95 % CI | Recall | 95 % CI | F2 | AP | 95 % CI | ROC-AUC | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stack | 0.864 | 0.756 – 0.944 | 0.986 | 0.962 – 1.000 | 0.959 | 0.893 | 0.771 – 0.991 | 0.680 | 0.133 | 0.431 |
| **LightGBM (shipped)** | 0.867 | 0.757 – 0.949 | 0.967 | 0.919 – 1.000 | 0.945 | 0.978 | 0.949 – 0.995 | 0.893 | 0.114 | 0.326 |
| ExtraTrees | 0.869 | 0.765 – 0.950 | 0.953 | 0.871 – 1.000 | 0.935 | 0.962 | 0.915 – 0.993 | 0.844 | 0.191 | 0.564 |
| RBF SVM | 0.842 | 0.731 – 0.934 | 0.941 | 0.856 – 1.000 | 0.919 | 0.915 | 0.816 – 0.986 | 0.703 | 0.250 | 0.900 |
| Logistic regression | 0.860 | 0.738 – 0.949 | 0.837 | 0.714 – 0.956 | 0.841 | 0.894 | 0.776 – 0.988 | 0.650 | 0.275 | 1.516 |
| Probability average | 0.868 | 0.757 – 0.955 | 0.882 | 0.770 – 0.979 | 0.879 | 0.951 | 0.891 – 0.994 | 0.799 | 0.171 | 0.511 |

Pooled held-out confusion matrix for `lightgbm`: 68 true positives, 11 false positives, 3 false negatives, 5 true negatives.

### Saved operating point

| | |
| --- | --- |
| Shipped model | `lightgbm` |
| Threshold | 0.184683 |
| Recall floor | 95% |
| Floor met on cross-fitted training rows | yes |
| Cross-fitted precision | 0.896 |
| Cross-fitted recall | 0.986 |
| Cross-fitted average precision | 0.975 |

The threshold is fitted on training stars. Pooled held-out recall for the shipped model was 0.967 (CI 0.919 – 1.000).

### Most important features

Permutation importance for `lightgbm` on held-out outer-fold stars, as the drop in accepted-candidate average precision; mean over folds and repeats, ± the standard deviation of the per-fold means.

| Feature | Importance | ± |
| --- | --- | --- |
| `MES` | 0.0383 | 0.0207 |
| `secondary_depth_snr` | 0.0125 | 0.0147 |
| `vshape_metric` | 0.0081 | 0.0101 |
| `max_mes` | 0.0020 | 0.0050 |
| `odd_even_depth_ratio` | 0.0019 | 0.0078 |
| `acf_lag_12h` | 0.0011 | 0.0024 |
| `secondary_depth` | 0.0002 | 0.0023 |
| `acf_lag_24h` | 0.0001 | 0.0035 |
| `acf_lag_1h` | 0.0001 | 0.0001 |
| `max_ses` | 0.0000 | 0.0000 |

Where the fold-to-fold spread exceeds the mean, that feature's importance is not distinguishable from noise across held-out stars; read the table as a weak ordering rather than an attribution.

Plots for calibration, precision-recall, ROC, score separation, permutation importance, and the model comparison are in `20260824T181710Z-256238b4/evaluation/plots/`.

## Limitations

- **Small, star-level sample.** The unit of independence is the star, not the
  candidate, so the effective sample size is the number of stars. Confidence
  intervals are computed by resampling stars and are correspondingly wide.
- **Label provenance.** Targets come from literature status, which is itself the
  output of a long human vetting process. The model learns what that process
  concluded, including its biases and its era.
- **Selection effects.** The training stars are ones with published dispositions.
  Candidates around quieter, brighter, or better-observed stars are
  over-represented relative to a blind survey.
- **Acceptance coupling.** The threshold and the meta-model are calibrated on
  accepted candidates only. Applying the score to candidates the detection
  pipeline rejected extrapolates outside the calibration set.
- **Pipeline dependence.** Features are computed by the [ltp-features](https://github.com/rachzy/ltp-features)
  detection pipeline. Feeding features produced by a different pipeline, or with
  different settings, invalidates the calibration even when the column names
  match.
- **Threshold optimism.** The 95 % recall floor is enforced on cross-fitted
  training predictions. It is a fitted quantity, and held-out recall may fall
  below it.
- **Selection optimism.** The shipped model is the best of six on a score that
  is then reported for it, so that score overstates expected performance. The
  effect grows with the number of candidates and shrinks with the number of
  stars; here there are six candidates and few dozen stars, so it is not
  negligible. A fully unbiased estimate would require selecting inside each
  outer fold, which this pipeline does not do.
- **Very little headroom at this operating point.** Accepted candidates are
  predominantly `CONFIRMED`, so a rule that simply flagged every accepted
  candidate would already reach 100 % recall at a precision equal to the base
  rate. Forcing 95 % recall on top of that leaves the model only a handful of
  false positives it can afford to exclude. Read the precision figures against
  the base rate stated in _Results_, never as absolute skill, and treat the gap
  between the two as the model's actual contribution.
- **Few negative examples to learn the boundary from.** The meta-model and the
  threshold are fitted on accepted candidates only, of which the `FALSE-POSITIVE`
  class is the smaller part. Both the operating point and the learned blend rest
  on a small number of negatives and are correspondingly unstable, which the
  star-bootstrap intervals reflect.

## Reproducibility

Every training run writes an immutable `models/<run-id>/` directory containing
the serialized stack and preprocessing pipelines, the threshold, the feature
schema, the resolved configuration, nested-CV metrics and out-of-fold
predictions, permutation importance and plots, the training-star list, input
file checksums, the Git commit, dependency versions, the seed, the run id, and a
`MANIFEST.json` with a SHA-256 for every file.

Tuning candidate sets, fold assignments, and bootstrap resamples are all derived
from the configured seed. Reloading a bundle reproduces probabilities and
decisions exactly; this is asserted in the test suite.

Bundles are Joblib pickles. Load only bundles produced by your own pipeline,
from a location you trust.

## Ethical and scientific considerations

The consequential failure here is not a wrong number, it is a screening score
being read as a discovery. The output column is named `potential_probability`
and the decision label is `POTENTIAL` rather than anything resembling
"confirmed" or "planet", the CLI restates the caveat on every prediction run,
and this card states it twice. Anyone publishing or acting on these scores
should cite the model's provenance (`model_run_id` is written into every
prediction row) and treat the flag as a request for attention, not a result.
