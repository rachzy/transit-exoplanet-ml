"""Period-matching helpers for grading model predictions against catalogs."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

# Recovered periods normally agree within about 0.01%; this remains tolerant
# of a poor fit while staying much tighter than typical planet spacing.
PERIOD_MATCH_TOLERANCE = 0.02

# Transit searches can lock onto small integer aliases of the true period.
ALIAS_MAX_NUMERATOR = 5
ALIAS_MAX_DENOMINATOR = 5

CONFIRMED_STEM_SEPARATORS = "-_. "
CONFIRMED_STEM_MARKERS = ("confirm", "confimed")


def _is_confirmed_stem_for(stem: str, star_name: str) -> bool:
    """Return whether a filename stem identifies this star's confirmed table."""
    stem = stem.strip()
    if len(stem) <= len(star_name):
        return False
    if stem[: len(star_name)].casefold() != star_name.casefold():
        return False

    remainder = stem[len(star_name) :]
    if remainder[0] not in CONFIRMED_STEM_SEPARATORS:
        return False
    folded = remainder.casefold()
    return any(marker in folded for marker in CONFIRMED_STEM_MARKERS)


def _alias_ratios() -> list[Fraction]:
    """Return supported period ratios, ordered from direct to mild aliases."""
    ratios = {
        Fraction(numerator, denominator)
        for numerator in range(1, ALIAS_MAX_NUMERATOR + 1)
        for denominator in range(1, ALIAS_MAX_DENOMINATOR + 1)
    }
    ratios.discard(Fraction(1))
    return [
        Fraction(1),
        *sorted(ratios, key=lambda ratio: (abs(np.log(float(ratio))), float(ratio))),
    ]


def _ratio_label(ratio: Fraction) -> str:
    """Format an extracted/catalog period ratio for the comparison report."""
    if ratio == 1:
        return "direct"
    numerator, denominator = ratio.numerator, ratio.denominator
    if denominator == 1:
        return f"{numerator}P"
    if numerator == 1:
        return f"P/{denominator}"
    return f"{numerator}P/{denominator}"


def _period_column(rows: pd.DataFrame) -> np.ndarray:
    if "period_days" not in rows.columns:
        return np.full(len(rows), np.nan)
    return pd.to_numeric(rows["period_days"], errors="coerce").to_numpy(dtype=float)


def _row_label(rows: pd.DataFrame, index: int, fallback: str) -> str:
    if "target" in rows.columns:
        value = rows.iloc[index].get("target", np.nan)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return fallback


def match_candidate_rows(
    extracted_rows: pd.DataFrame,
    confirmed_rows: pd.DataFrame,
    *,
    tolerance: float = PERIOD_MATCH_TOLERANCE,
) -> list[dict]:
    """Pair extracted and confirmed candidates by direct or aliased period.

    Each row is used at most once. The result also contains unmatched catalog
    rows (``missed``) and unmatched extracted rows (``extra``).
    """
    extracted_periods = _period_column(extracted_rows)
    confirmed_periods = _period_column(confirmed_rows)
    ratios = _alias_ratios()

    pairings = []
    for extracted_index, extracted_period in enumerate(extracted_periods):
        if not (np.isfinite(extracted_period) and extracted_period > 0):
            continue
        for confirmed_index, confirmed_period in enumerate(confirmed_periods):
            if not (np.isfinite(confirmed_period) and confirmed_period > 0):
                continue
            for ratio in ratios:
                expected = float(ratio) * confirmed_period
                relative = abs(extracted_period - expected) / expected
                if relative <= tolerance:
                    pairings.append(
                        (
                            0 if ratio == 1 else 1,
                            relative,
                            extracted_index,
                            confirmed_index,
                            ratio,
                        )
                    )
                    break
    pairings.sort()

    matches: list[dict] = []
    claimed_extracted: set[int] = set()
    claimed_confirmed: set[int] = set()
    for _priority, relative, extracted_index, confirmed_index, ratio in pairings:
        if extracted_index in claimed_extracted or confirmed_index in claimed_confirmed:
            continue
        claimed_extracted.add(extracted_index)
        claimed_confirmed.add(confirmed_index)
        matches.append(
            {
                "kind": "direct" if ratio == 1 else "alias",
                "extracted_index": extracted_index,
                "confirmed_index": confirmed_index,
                "ratio": float(ratio),
                "ratio_label": _ratio_label(ratio),
                "period_rel_diff": float(relative),
                "target": _row_label(
                    confirmed_rows, confirmed_index, f"candidate-{confirmed_index + 1}"
                ),
                "extracted_period": float(extracted_periods[extracted_index]),
                "confirmed_period": float(confirmed_periods[confirmed_index]),
            }
        )

    for confirmed_index, confirmed_period in enumerate(confirmed_periods):
        if confirmed_index in claimed_confirmed:
            continue
        matches.append(
            {
                "kind": "missed",
                "extracted_index": None,
                "confirmed_index": confirmed_index,
                "ratio": float("nan"),
                "ratio_label": "missed",
                "period_rel_diff": float("nan"),
                "target": _row_label(
                    confirmed_rows, confirmed_index, f"candidate-{confirmed_index + 1}"
                ),
                "extracted_period": float("nan"),
                "confirmed_period": float(confirmed_period),
            }
        )

    for extracted_index, extracted_period in enumerate(extracted_periods):
        if extracted_index in claimed_extracted:
            continue
        matches.append(
            {
                "kind": "extra",
                "extracted_index": extracted_index,
                "confirmed_index": None,
                "ratio": float("nan"),
                "ratio_label": "unmatched",
                "period_rel_diff": float("nan"),
                "target": None,
                "extracted_period": float(extracted_period),
                "confirmed_period": float("nan"),
            }
        )

    order = {"direct": 0, "alias": 1, "missed": 2, "extra": 3}
    matches.sort(
        key=lambda match: (
            order[match["kind"]],
            match["confirmed_period"]
            if np.isfinite(match["confirmed_period"])
            else match["extracted_period"],
        )
    )
    return matches
