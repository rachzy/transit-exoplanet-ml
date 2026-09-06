#!/usr/bin/env python3
"""Grade a predictions CSV against the confirmed catalogs in ``data/literature``.

Every scored row is matched to a confirmed planet by orbital period (the same
alias-aware matcher the extraction comparison uses), which turns the model's
``POTENTIAL``/``UNLIKELY`` calls into a confusion matrix:

    true positive   confirmed planet the model called POTENTIAL
    false negative  confirmed planet the model called UNLIKELY
    true negative   row matching no catalog planet, called UNLIKELY
    false positive  row matching no catalog planet, called POTENTIAL

Confirmed planets that never reached the predictions file at all are reported
separately as ``not found in preprocessing`` and are excluded from every count:
the model cannot be graded on a candidate it was never shown.

Usage (with the project venv active)::

    python src/scripts/compare_confirmed_and_prediction.py
    python src/scripts/compare_confirmed_and_prediction.py --predictions runs/p.csv
    python src/scripts/compare_confirmed_and_prediction.py --threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.compare_extracted_confirmed import (  # noqa: E402
    PERIOD_MATCH_TOLERANCE,
    _is_confirmed_stem_for,
    match_candidate_rows,
)

# Mirrors src/training.POTENTIAL: importing it would pull the whole modelling
# stack in just to read one string.
POTENTIAL = "POTENTIAL"

TRUE_POSITIVE = "true_positive"
FALSE_NEGATIVE = "false_negative"
TRUE_NEGATIVE = "true_negative"
FALSE_POSITIVE = "false_positive"
NOT_FOUND = "not_found_in_preprocessing"

# Order used for both the per-category listings and the final tally.
CATEGORIES = (
    (TRUE_POSITIVE, "TRUE POSITIVES", "confirmed planet, called POTENTIAL"),
    (TRUE_NEGATIVE, "TRUE NEGATIVES", "no catalog match, called UNLIKELY"),
    (FALSE_NEGATIVE, "FALSE NEGATIVES", "confirmed planet, called UNLIKELY"),
    (FALSE_POSITIVE, "FALSE POSITIVES", "no catalog match, called POTENTIAL"),
    (
        NOT_FOUND,
        "NOT FOUND IN PREPROCESSING",
        "confirmed planet absent from the predictions file - not graded",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade a predictions CSV against confirmed catalogs and report "
            "true/false positives and negatives per candidate."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO_ROOT / "predictions.csv",
        help="Predictions CSV written by `predict` (default: predictions.csv)",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "literature",
        help="Directory of confirmed CSVs (default: data/literature)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Re-decide every row at this probability threshold instead of "
            "using the `prediction` column recorded at prediction time"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=PERIOD_MATCH_TOLERANCE,
        help=(
            "Relative period agreement required to call a row the same planet "
            f"(default: {PERIOD_MATCH_TOLERANCE})"
        ),
    )
    return parser.parse_args(argv)


def find_confirmed_csv(star: str, confirmed_dir: Path) -> Path | None:
    """Locate ``star``'s confirmed table without touching the network.

    Only a file naming this exact star is accepted: ``Kepler-42`` is a prefix
    of ``Kepler-421``, and grading a star against its neighbour's periods turns
    every real planet into a false negative.
    """
    if not confirmed_dir.is_dir():
        return None
    for name in (f"{star}-confirmed.csv", f"{star}-confimed.csv"):
        path = confirmed_dir / name
        if path.is_file():
            return path
    matches = sorted(
        p for p in confirmed_dir.iterdir() if p.is_file() and _is_confirmed_stem_for(p.stem, star)
    )
    return matches[0] if matches else None


def _as_float(value) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if pd.notna(number) else float("nan")


def _is_potential(row: pd.Series, threshold: float | None) -> bool:
    """Did the model call this row a planet?

    The recorded ``prediction`` column is authoritative unless the caller asked
    for a different threshold, in which case the decision is recomputed from
    the stored probability.
    """
    if threshold is None and "prediction" in row.index:
        return str(row["prediction"]).strip().upper() == POTENTIAL
    probability = _as_float(row.get("potential_probability", np.nan))
    if not np.isfinite(probability):
        raise ValueError(
            "Cannot decide this row: no usable `prediction` or "
            "`potential_probability` column in the predictions CSV."
        )
    cutoff = threshold
    if cutoff is None:
        cutoff = _as_float(row.get("decision_threshold", np.nan))
    if not np.isfinite(cutoff):
        raise ValueError("No decision threshold available; pass --threshold.")
    return probability >= cutoff


def grade_star(
    star: str,
    predicted_rows: pd.DataFrame,
    confirmed_rows: pd.DataFrame,
    *,
    threshold: float | None,
    tolerance: float,
) -> list[dict]:
    """Assign every row of one star to a confusion-matrix cell."""
    predicted_rows = predicted_rows.reset_index(drop=True)
    confirmed_rows = confirmed_rows.reset_index(drop=True)
    graded: list[dict] = []

    for match in match_candidate_rows(predicted_rows, confirmed_rows, tolerance=tolerance):
        kind = match["kind"]
        if kind == "missed":
            graded.append(
                {
                    "category": NOT_FOUND,
                    "star": star,
                    "candidate": match["target"],
                    "catalog_period": match["confirmed_period"],
                    "predicted_period": float("nan"),
                    "match": "-",
                    "probability": float("nan"),
                    "call": "-",
                    "status": "-",
                }
            )
            continue

        row = predicted_rows.iloc[match["extracted_index"]]
        potential = _is_potential(row, threshold)
        confirmed = kind in ("direct", "alias")
        if confirmed:
            category = TRUE_POSITIVE if potential else FALSE_NEGATIVE
            candidate = match["target"]
        else:
            category = FALSE_POSITIVE if potential else TRUE_NEGATIVE
            candidate = f"{star} P={match['extracted_period']:.4f}d"
        graded.append(
            {
                "category": category,
                "star": star,
                "candidate": candidate,
                "catalog_period": match["confirmed_period"],
                "predicted_period": match["extracted_period"],
                "match": match["ratio_label"],
                "probability": _as_float(row.get("potential_probability", np.nan)),
                "call": POTENTIAL if potential else "UNLIKELY",
                "status": _detection_status(row),
            }
        )
    return graded


def _detection_status(row: pd.Series) -> str:
    value = row.get("detection_status", None)
    if value is None or pd.isna(value) or not str(value).strip():
        return "-"
    return str(value).strip()


def _format_period(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def _format_probability(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.4f}"


def format_category_table(records: list[dict], category: str) -> pd.DataFrame:
    """Render one confusion-matrix cell as a printable table."""
    rows = [r for r in records if r["category"] == category]

    def sort_key(record: dict) -> tuple[str, float]:
        probability = record["probability"]
        # Highest-scoring candidate first; unscored rows sort last.
        return record["star"], -(probability if np.isfinite(probability) else -1.0)

    rows.sort(key=sort_key)
    display = pd.DataFrame(
        [
            {
                "star": r["star"],
                "candidate": r["candidate"],
                "pred_period": _format_period(r["predicted_period"]),
                "catalog_period": _format_period(r["catalog_period"]),
                "match": r["match"],
                "probability": _format_probability(r["probability"]),
                "call": r["call"],
                "status": r["status"],
            }
            for r in rows
        ]
    )
    if display.empty:
        return display
    return display.set_index("star")


def summarize(records: list[dict]) -> dict[str, int]:
    counts = {key: 0 for key, _title, _blurb in CATEGORIES}
    for record in records:
        counts[record["category"]] += 1
    return counts


def format_metrics(counts: dict[str, int]) -> list[str]:
    """Precision/recall/F1/accuracy over the graded rows only."""
    tp = counts[TRUE_POSITIVE]
    fn = counts[FALSE_NEGATIVE]
    tn = counts[TRUE_NEGATIVE]
    fp = counts[FALSE_POSITIVE]
    graded = tp + fn + tn + fp

    def ratio(numerator: int, denominator: int) -> str:
        return "n/a" if denominator == 0 else f"{numerator / denominator:.4f}"

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return [
        f" graded rows   : {graded}",
        f" precision     : {ratio(tp, tp + fp)}",
        f" recall        : {ratio(tp, tp + fn)}",
        f" f1            : {'n/a' if not np.isfinite(f1) else f'{f1:.4f}'}",
        f" accuracy      : {ratio(tp + tn, graded)}",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.predictions.is_file():
        print(f"Predictions file not found: {args.predictions}", file=sys.stderr)
        return 1
    if not args.confirmed_dir.is_dir():
        print(f"Confirmed directory not found: {args.confirmed_dir}", file=sys.stderr)
        return 1

    predictions = pd.read_csv(args.predictions)
    if predictions.empty:
        print(f"{args.predictions} has no data rows.")
        return 0
    if "star_id" not in predictions.columns:
        print(f"{args.predictions} has no `star_id` column.", file=sys.stderr)
        return 1

    print(f"Grading {len(predictions)} scored row(s) from {args.predictions}")
    if args.threshold is not None:
        print(f"Re-deciding every row at threshold {args.threshold}")
    print()

    records: list[dict] = []
    no_catalog: dict[str, int] = {}

    for star, star_rows in predictions.groupby("star_id", sort=True):
        confirmed_path = find_confirmed_csv(str(star), args.confirmed_dir)
        if confirmed_path is None:
            no_catalog[str(star)] = len(star_rows)
            continue
        print(f"  {star}: {len(star_rows)} row(s) ↔ {confirmed_path.name}")
        records.extend(
            grade_star(
                str(star),
                star_rows,
                pd.read_csv(confirmed_path),
                threshold=args.threshold,
                tolerance=args.tolerance,
            )
        )

    if no_catalog:
        listed = ", ".join(f"{star} ({n} row(s))" for star, n in sorted(no_catalog.items()))
        print(f"\nNo confirmed catalog - ungraded ({len(no_catalog)}): {listed}")

    if not records:
        print("\nNothing to grade.")
        return 0

    for category, title, blurb in CATEGORIES:
        table = format_category_table(records, category)
        print()
        print("=" * 78)
        print(f" {title}  ({len(table)})")
        print(f" {blurb}")
        print("=" * 78)
        print(table.to_string() if not table.empty else " none")

    counts = summarize(records)
    print()
    print("=" * 78)
    print(" TOTALS")
    print("=" * 78)
    for category, title, _blurb in CATEGORIES:
        print(f" {title.lower():<28}: {counts[category]}")
    if no_catalog:
        print(f" {'ungraded (no catalog)':<28}: {sum(no_catalog.values())}")
    print("-" * 78)
    for line in format_metrics(counts):
        print(line)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
