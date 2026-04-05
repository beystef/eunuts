"""
UN Comtrade - Monthly Bilateral Export Fetcher for Nuts and Seeds (2000-2023)
=============================================================================
Uses : comtradeapicall.getFinalData()
Data : Monthly bilateral exports (all reporters x all partners)
Scope: 11 HS6 products, 2000-2023

Products covered
----------------
  080221 : Hazelnuts, in shell
  080222 : Hazelnuts, shelled
  080211 : Almonds, in shell
  080212 : Almonds, shelled
  080231 : Walnuts, in shell
  080232 : Walnuts, shelled
  080251 : Pistachios, in shell
  080252 : Pistachios, shelled
  120241 : Peanuts, in shell
  120242 : Peanuts, shelled
  120740 : Sesame seeds

Output files
------------
  data/raw/<HS6_CODE>_<YEAR>.csv   - raw bilateral monthly export data per year
  data/<HS6_CODE>_all.csv          - combined 2000-2023 dataset for each HS6 code

Install
-------
  pip install comtradeapicall pandas python-dotenv

API key
-------
  Register at https://comtradedeveloper.un.org and subscribe to "comtrade - v1"
  Set env var: COMTRADE_KEY="your_key_here"
"""

import logging
import os
import time
from pathlib import Path

import comtradeapicall
import pandas as pd
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


load_dotenv()

SUBSCRIPTION_KEY: str = os.getenv("COMTRADE_KEY", "NO_SUBSCRIPTION_KEY")
YEARS = list(range(2000, 2024))
FLOW_CODE = "X"
MAX_RECORDS = 250_000
DELAY_SEC = 5
OUTPUT_DIR = Path("data")
RAW_DIR = OUTPUT_DIR / "raw"

HS6_CODES = [
    "080221",
    "080222",
    "080211",
    "080212",
    "080231",
    "080232",
    "080251",
    "080252",
    "120241",
    "120242",
    "120740",
]

KEEP_COLS = [
    "typeCode",
    "freqCode",
    "refYear",
    "refMonth",
    "period",
    "reporterCode",
    "reporterISO",
    "reporterDesc",
    "partnerCode",
    "partnerISO",
    "partnerDesc",
    "flowCode",
    "flowDesc",
    "cmdCode",
    "cmdDesc",
    "primaryValue",
    "netWgt",
    "qty",
    "qtyUnitAbbr",
]


def make_period_string(year: int) -> str:
    """Return a comma-separated YYYYMM string for all 12 months of a year."""
    return ",".join(f"{year}{month:02d}" for month in range(1, 13))


def fetch_year(hs6_code: str, year: int) -> pd.DataFrame:
    """Fetch one year of monthly bilateral export data for a single HS6 code."""
    period_str = make_period_string(year)
    log.info("  Fetching HS6 %s | %s", hs6_code, year)

    try:
        df = comtradeapicall.getFinalData(
            subscription_key=SUBSCRIPTION_KEY,
            typeCode="C",
            freqCode="M",
            clCode="HS",
            period=period_str,
            reporterCode=None,
            cmdCode=hs6_code,
            flowCode=FLOW_CODE,
            partnerCode=None,
            partner2Code=None,
            customsCode=None,
            motCode=None,
            maxRecords=MAX_RECORDS,
            format_output="JSON",
            aggregateBy=None,
            breakdownMode="classic",
            countOnly=None,
            includeDesc=True,
        )
    except Exception as exc:
        log.error("    API error for HS6 %s / %s: %s", hs6_code, year, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        log.warning("    No data returned for HS6 %s / %s", hs6_code, year)
        return pd.DataFrame()

    log.info("    -> %s rows", f"{len(df):,}")

    df["hs6"] = hs6_code
    available_cols = [col for col in KEEP_COLS if col in df.columns] + ["hs6"]
    return df[available_cols]


def save_year_raw(df: pd.DataFrame, hs6_code: str, year: int) -> None:
    """Save one raw year file for one HS6 code."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{hs6_code}_{year}.csv"
    df.to_csv(path, index=False)
    log.info("    Saved -> %s", path)


def save_code_all(df_list: list[pd.DataFrame], hs6_code: str) -> None:
    """Combine and save all years for one HS6 code."""
    combined = pd.concat(df_list, ignore_index=True)
    path = OUTPUT_DIR / f"{hs6_code}_all.csv"
    combined.to_csv(path, index=False)
    log.info("  Saved combined file for HS6 %s -> %s (%s rows)", hs6_code, path, f"{len(combined):,}")


def main() -> None:
    if SUBSCRIPTION_KEY == "NO_SUBSCRIPTION_KEY":
        log.error(
            "No API key set. Set COMTRADE_KEY=<your_key> in your environment or .env file.\n"
            "Get a key at: https://comtradedeveloper.un.org"
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_calls = len(HS6_CODES) * len(YEARS)
    call_num = 0

    log.info(
        "Starting export fetch: %s HS6 codes x %s years = %s API calls",
        len(HS6_CODES),
        len(YEARS),
        total_calls,
    )

    for hs6_code in HS6_CODES:
        log.info("\n%s", "━" * 60)
        log.info("HS6 code: %s", hs6_code)
        log.info("%s", "━" * 60)

        code_frames: list[pd.DataFrame] = []

        for year in YEARS:
            call_num += 1
            log.info("[%s/%s]", call_num, total_calls)

            df = fetch_year(hs6_code, year)

            if not df.empty:
                save_year_raw(df, hs6_code, year)
                code_frames.append(df)

            time.sleep(DELAY_SEC)

        if code_frames:
            save_code_all(code_frames, hs6_code)
        else:
            log.warning("No data collected for HS6 %s across all years.", hs6_code)


if __name__ == "__main__":
    main()
