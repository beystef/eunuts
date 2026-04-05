"""
Filter yearly raw Comtrade export files and save cleaned yearly panels.

Input:
  data/raw/<HS6>_<YEAR>.csv

Output:
  data/processed/<HS6>_<YEAR>.csv

Filtering strategy
------------------
Stage 1: Hard filters
  - Drop rows with missing or nonpositive net weight
  - Drop rows with missing or nonpositive trade value
  - Drop rows with implausibly small quantities (< 1 kg)

Stage 2: Log-MAD trimming within (cmdCode, refYear)
  - Compute UnitValue = primaryValue / netWgt
  - Compute LogUnitValue = log(UnitValue)
  - Keep observations within median +/- 3 * MAD of LogUnitValue

Saved columns
-------------
  year_month, reporterISO, partnerISO, cmdCode,
  primaryValue, NetWeight, UnitValue, LogUnitValue
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_COLUMNS = [
    "year_month",
    "reporterISO",
    "partnerISO",
    "cmdCode",
    "primaryValue",
    "NetWeight",
    "UnitValue",
    "LogUnitValue",
]


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load one yearly raw CSV file."""
    df = pd.read_csv(path)
    df["source_file"] = path.name
    return df


def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Remove structurally invalid observations before outlier trimming."""
    work = df.copy()

    work["primaryValue"] = pd.to_numeric(work["primaryValue"], errors="coerce")
    work["netWgt"] = pd.to_numeric(work["netWgt"], errors="coerce")
    work["reporterISO"] = work["reporterISO"].astype(str)
    work["partnerISO"] = work["partnerISO"].astype(str)

    keep_mask = (
        work["reporterISO"].ne("W00")
        & work["partnerISO"].ne("W00")
        & work["primaryValue"].notna()
        & work["netWgt"].notna()
        & work["primaryValue"].gt(0)
        & work["netWgt"].gt(0)
        & work["netWgt"].ge(1)
    )

    return work.loc[keep_mask].copy()


def mad_trim_group(group: pd.DataFrame) -> pd.DataFrame:
    """Trim outliers in one (cmdCode, refYear) group using log-MAD."""
    work = group.copy()
    median = work["LogUnitValue"].median()
    mad = (work["LogUnitValue"] - median).abs().median()

    if pd.isna(mad):
        return work.iloc[0:0].copy()

    if mad == 0:
        keep_mask = work["LogUnitValue"].eq(median)
        return work.loc[keep_mask].copy()

    lower = median - 3 * mad
    upper = median + 3 * mad
    keep_mask = work["LogUnitValue"].between(lower, upper, inclusive="both")
    return work.loc[keep_mask].copy()


def apply_log_mad_trim(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log-MAD trimming within each (cmdCode, refYear) stratum."""
    work = df.copy()
    if work.empty:
        work["UnitValue"] = pd.Series(dtype="float64")
        work["LogUnitValue"] = pd.Series(dtype="float64")
        return work

    work["UnitValue"] = work["primaryValue"] / work["netWgt"]
    work["LogUnitValue"] = np.log(work["UnitValue"])

    trimmed_groups: list[pd.DataFrame] = []
    for _, group in work.groupby(["cmdCode", "refYear"], dropna=False):
        trimmed_group = mad_trim_group(group)
        if not trimmed_group.empty:
            trimmed_groups.append(trimmed_group)

    if not trimmed_groups:
        return work.iloc[0:0].copy()

    return pd.concat(trimmed_groups, ignore_index=True)


def format_output(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the requested output columns and rename them."""
    work = df.copy()
    if work.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    work["cmdCode"] = work["cmdCode"].astype(str).str.zfill(6)
    work["reporterISO"] = work["reporterISO"].astype(str)
    work["partnerISO"] = work["partnerISO"].astype(str)
    work["year_month"] = work["period"].astype(str)

    output = work[
        [
            "year_month",
            "reporterISO",
            "partnerISO",
            "cmdCode",
            "primaryValue",
            "netWgt",
            "UnitValue",
            "LogUnitValue",
        ]
    ].rename(columns={"netWgt": "NetWeight"})

    return output[OUTPUT_COLUMNS]


def process_file(path: Path) -> None:
    """Filter one yearly raw file and save the processed result."""
    log.info("Processing %s", path.name)
    raw_df = load_raw_file(path)
    filtered_df = apply_hard_filters(raw_df)
    trimmed_df = apply_log_mad_trim(filtered_df)
    output_df = format_output(trimmed_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / path.name
    output_df.to_csv(out_path, index=False)
    log.info("Saved %s rows to %s", f"{len(output_df):,}", out_path)


def main() -> None:
    raw_files = sorted(RAW_DIR.glob("*.csv"))

    if not raw_files:
        log.warning("No raw CSV files found in %s", RAW_DIR)
        return

    for path in raw_files:
        process_file(path)


if __name__ == "__main__":
    main()
