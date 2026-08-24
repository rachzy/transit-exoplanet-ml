# Model Card — Transit Exoplanet Candidate Screener

## Model details

| | |
| --- | --- |
| **Name** | transit-exoplanet-ml candidate screener |
| **Version** | 0.1.0 (feature schema version 1) |
| **Type** | Stacked binary classifier |
| **Base learners** | LightGBM, ExtraTrees (500 trees), RBF SVM (Platt-scaled), L2 logistic regression |
| **Meta-learner** | L2 logistic regression, `C = 0.1`, fixed and never tuned |
| **Meta inputs** | the four base probabilities, and nothing else |
| **Output** | `potential_probability` ∈ [0, 1] and a `POTENTIAL` / `UNLIKELY` decision |
| **Seed** | 42 |
| **License** | see `LICENSE` |

The production model is **always** this stack. If a single base learner or the
unweighted probability-average baseline scores higher, the stack still ships and
the underperformance is reported by the CLI and recorded in the artifact.

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
- Stars or instruments unlike those in the training set (see *Limitations*).

## Training data

Processed per-star candidate tables under `data/processed/train/`, one CSV per
star, named `<star-name>_<YYYYMMDD>.csv`. Each row is one candidate.

Two independent axes are used:

- **`candidate_label`** — the target: `CONFIRMED → 1`, `FALSE-POSITIVE → 0`.
- **`detection_status`** — whether the detection pipeline *accepted* the
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

Reported for the stack, every base learner, and an unweighted
probability-average baseline: precision and recall at the chosen threshold,
average precision, ROC-AUC, F2, confusion matrix, Brier score, log loss, a
calibration curve, per-star summaries, and star-bootstrap 95 % confidence
intervals.

## Results

Run `20260824T150116Z-4ad52e7f`, seed 42, schema version 1.

### Training set

| | |
| --- | --- |
| Stars | 35 |
| Candidates | 228 |
| Accepted candidates | 87 (67 CONFIRMED / 20 FALSE-POSITIVE) |
| Non-accepted candidates | 141 (base-learner training only) |
| Model features | 36 |

Accepted candidates are **77% CONFIRMED**. That is the precision a "flag everything" rule would achieve at 100 % recall, and it is the number every precision below should be read against.

### Nested evaluation

5 outer folds, inner folds [3, 3, 3, 3, 3], grouped by star. Each fold picked its own threshold from its outer-training predictions only. Intervals are percentile 95 % CIs from 1000 star-level bootstrap resamples.

| Model | Precision | 95 % CI | Recall | 95 % CI | F2 | AP | 95 % CI | ROC-AUC | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Stack (production)** | 0.829 | 0.697 – 0.932 | 1.000 | 1.000 – 1.000 | 0.960 | 0.885 | 0.748 – 0.979 | 0.659 | 0.153 | 0.480 |
| LightGBM | 0.840 | 0.701 – 0.949 | 0.933 | 0.842 – 1.000 | 0.913 | 0.887 | 0.730 – 0.990 | 0.719 | 0.169 | 0.584 |
| ExtraTrees | 0.829 | 0.693 – 0.937 | 0.933 | 0.842 – 1.000 | 0.911 | 0.864 | 0.719 – 0.982 | 0.654 | 0.220 | 0.636 |
| RBF SVM | 0.816 | 0.682 – 0.914 | 0.962 | 0.917 – 1.000 | 0.929 | 0.891 | 0.785 – 0.963 | 0.654 | 0.307 | 0.827 |
| Logistic regression | 0.796 | 0.667 – 0.901 | 0.935 | 0.837 – 1.000 | 0.904 | 0.793 | 0.642 – 0.973 | 0.571 | 0.262 | 2.494 |
| Probability average (baseline) | 0.842 | 0.713 – 0.944 | 0.922 | 0.806 – 1.000 | 0.905 | 0.915 | 0.800 – 0.985 | 0.746 | 0.182 | 0.541 |

Pooled held-out confusion matrix for the stack: 67 true positives, 17 false positives, 0 false negatives, 3 true negatives.

On the metric the objective actually targets, the stack has the best F2 (0.960) and the highest recall (1.000) of any model reported here.

**It is nevertheless outperformed on threshold-free ranking**: average precision 0.885 against Probability average (baseline) (0.915), RBF SVM (0.891), LightGBM (0.887). The stack ships regardless, as specified. Two things drive this, and both are consequences of a small meta-training set rather than of a defect:

1. The meta-model is fitted on the out-of-fold base probabilities of 87 accepted candidates. Four correlated inputs and that many rows are not enough to learn reliably which base learner to trust, so the learned blend generalises worse than the unweighted average of the same four probabilities.
2. The blend necessarily carries weight on the weakest base learner (logistic regression, AP 0.793), which a single strong learner does not.

The practical reading: the stack is the safer choice at the high-recall operating point it was tuned for, and the weaker choice if the scores are used to rank candidates rather than to threshold them.

### Saved operating point

| | |
| --- | --- |
| Threshold | 0.707237 |
| Recall floor | 95% |
| Floor met on cross-fitted training rows | yes |
| Cross-fitted precision | 0.810 |
| Cross-fitted recall | 1.000 |
| Cross-fitted average precision | 0.888 |

The threshold is fitted on training stars. Pooled held-out recall for the stack was 1.000 (CI 1.000 – 1.000).

### Most important features

Permutation importance on held-out outer-fold stars, as the drop in accepted-candidate average precision; mean over folds and repeats, ± the standard deviation of the per-fold means.

| Feature | Importance | ± |
| --- | --- | --- |
| `MES` | 0.0178 | 0.0296 |
| `planet_radius_rearth` | 0.0140 | 0.0203 |
| `period_days` | 0.0104 | 0.0140 |
| `vshape_metric` | 0.0094 | 0.0322 |
| `skewness_flux` | 0.0086 | 0.0072 |
| `max_mes` | 0.0076 | 0.0102 |
| `SES_mean` | 0.0066 | 0.0083 |
| `acf_lag_3h` | 0.0058 | 0.0092 |
| `acf_lag_12h` | 0.0058 | 0.0058 |
| `secondary_depth_snr` | 0.0054 | 0.0076 |

For most features the fold-to-fold spread exceeds the mean, so **no single
feature is robustly important** across held-out stars. Read this table as a weak
ordering, not as an attribution: the signal is spread thinly across the
transit-significance and geometry features rather than concentrated in any one
of them.

A second observation worth recording: the stack's cross-fitted probabilities all
fall in a narrow band, roughly 0.65 – 0.90, which is what the score-separation
plot shows. An L2 logistic meta-model at `C = 0.1` reading four unscaled
probabilities is strongly shrunk toward its intercept, so `potential_probability`
compresses. Ranking survives the compression, but the values should not be read
as calibrated confidences; the calibration curve in the plots directory is the
honest view.

Plots for calibration, precision-recall, ROC, score separation, permutation importance, and the model comparison are in `20260824T150116Z-4ad52e7f/evaluation/plots/`.

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
- **Pipeline dependence.** Features are computed by a specific upstream
  detection pipeline. Feeding features produced by a different pipeline, or with
  different settings, invalidates the calibration even when the column names
  match.
- **Threshold optimism.** The 95 % recall floor is enforced on cross-fitted
  training predictions. It is a fitted quantity, and held-out recall may fall
  below it.
- **Very little headroom at this operating point.** Accepted candidates are
  predominantly `CONFIRMED`, so a rule that simply flagged every accepted
  candidate would already reach 100 % recall at a precision equal to the base
  rate. Forcing 95 % recall on top of that leaves the model only a handful of
  false positives it can afford to exclude. Read the precision figures against
  the base rate stated in *Results*, never as absolute skill, and treat the gap
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
