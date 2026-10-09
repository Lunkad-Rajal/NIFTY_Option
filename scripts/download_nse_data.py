"""
download of NSE F&O Bhavcopy files into data/raw/.

Usage:
    python scripts/download_nse_data.py --start 2026-08-01 --end 2026-09-30

If a download fails, it logs the failure and continues. 
Files that succeed are saved with the names the pipeline expects, so a later run of:

    python -m nifty_opv.cli --config config.yaml

will pick them up automatically from data/raw/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("download_nse_data")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _weekdays(start: date, end: date) -> list[date]:
    """Return all weekdays between start and end, inclusive."""
    days: list[date] = []
    cur = start
    while cur <= end:
        # 0=Mon..6=Sun; NSE is closed on weekends.
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days


def _save(df: pd.DataFrame, target: Path) -> None:
    """Save one day's data as a CSV that datapipe.py will parse."""
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    logger.info("Saved %s (%d rows)", target, len(df))


def _try_nse_archives(target_date: date) -> pd.DataFrame | None:
    """
    fetch one day's F&O Bhavcopy via the nse-archives package.

    Returns a DataFrame on success, ``None`` on any failure (import error,
    API change, HTTP error, empty result).
    """
    try:
        from nsedata import nse  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("nse-archives is not installed. Run: pip install nse-archives")
        return None

    # Try the documented UDiFF call. If the API has changed, this raises an exception which we log and treat as a failure.
    try:
        df = nse.get(
            "derivatives",
            "equity",
            "fo_bhav_udiff",
            target_date.isoformat(),
        )
    except Exception as exc:
        logger.warning("nse-archives call failed for %s: %s", target_date, exc)
        return None

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("nse-archives returned nothing for %s", target_date)
        return None

    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download NSE F&O Bhavcopy files.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--out", default="data/raw", help="Output directory")
    args = p.parse_args(argv)

    start = _parse_date(args.start)
    end = _parse_date(args.end)
    if end < start:
        logger.error("--end must be on or after --start")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = _weekdays(start, end)
    logger.info("Attempting download for %d weekdays between %s and %s",
                len(days), start, end)

    n_ok = 0
    n_fail = 0
    for d in days:
        target = out_dir / f"nse_fo_{d.strftime('%Y%m%d')}.csv"
        if target.exists():
            logger.info("Already present: %s", target)
            n_ok += 1
            continue

        df = _try_nse_archives(d)
        if df is None:
            n_fail += 1
            continue

        try:
            _save(df, target)
            n_ok += 1
        except OSError as exc:
            logger.warning("Could not write %s: %s", target, exc)
            n_fail += 1

    logger.info("Download complete: %d files ok, %d failed", n_ok, n_fail)

    if n_ok == 0:
        logger.error("")
        logger.error("No files were saved. Two paths forward:")
        logger.error("")
        logger.error("  1. Inspect the nse-archives docs for your installed version:")
        logger.error("     python -c \"import nsedata; help(nsedata.nse)\"")
        logger.error("")
        logger.error("  2. Download manually from:")
        logger.error("     https://www.nseindia.com/all-reports-derivatives")
        logger.error("     Place each day's ZIP or CSV into data/raw/.")
        logger.error("     The pipeline reads any of these suffixes:")
        logger.error("     .dat, .csv, .zip, .gz, .txt")
        logger.error("")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
