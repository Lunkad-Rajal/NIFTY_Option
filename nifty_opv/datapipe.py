"""
Data acquisition, parsing, normalization, and cleaning for NSE NIFTY options.

- :func:`download_bhavcopy` is best-effort. NSE blocks scrapers, and the
  pipeline must work when this fails. Failures return None and are
  logged; nothing is fabricated.
- :func:`parse_bhavcopy` handles the file formats NSE has shipped over
  time. Anything it cannot parse raises a clear error rather than
  silently producing an empty frame.
- :func:`normalize_columns` maps source column names to a canonical schema
  and translates NSE UDiFF instrument-type codes to canonical instrument
  types (OPTIDX, FUTIDX, OPTSTK, FUTSTK).
- :func:`clean_options_data` applies config filters and returns a
  :class:`DataQualityReport` with per-reason exclusion counts.

Canonical schema (all lower-case, snake_case)
---------------------------------------------
trade_date, underlying_symbol, instrument_type, expiry_date, strike,
option_type, open, high, low, close, settlement_price, volume,
open_interest, underlying_spot, source_file

Missing fields are NaN, not zero.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nifty_opv.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "DataQualityReport",
    "download_bhavcopy",
    "parse_bhavcopy",
    "normalize_columns",
    "clean_options_data",
    "load_spot_prices",
    "load_all_raw",
    "CANONICAL_COLUMNS",
    "UDIFF_INSTRUMENT_TYPE_MAP",
]

CANONICAL_COLUMNS: list[str] = [
    "trade_date",
    "underlying_symbol",
    "instrument_type",
    "expiry_date",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "settlement_price",
    "volume",
    "open_interest",
    "underlying_spot",
    "source_file",
]


# NSE UDiFF instrument-type translation

#
UDIFF_INSTRUMENT_TYPE_MAP: dict[str, str] = {
    "IDO": "OPTIDX",
    "IDF": "FUTIDX",
    "STO": "OPTSTK",
    "STF": "FUTSTK",
    # Legacy / alternate names that occasionally appear in older archives.
    "OPTIDX": "OPTIDX",
    "FUTIDX": "FUTIDX",
    "OPTSTK": "OPTSTK",
    "FUTSTK": "FUTSTK",
    "OPTIONS": "OPTIDX",
    "FUTURES": "FUTIDX",
}


# Data quality report


@dataclass
class DataQualityReport:
    """
    Human-readable record of every row that entered and left the pipeline.

    Every exclusion path must increment one of the counters below. There
    should never be an "excluded_rows" that is not also explained by a
    named reason.
    """

    input_rows: int = 0
    output_rows: int = 0
    excluded_rows: int = 0
    duplicates_removed: int = 0
    zero_volume_excluded: int = 0
    zero_open_interest_count: int = 0
    missing_spot_count: int = 0
    excluded_by_reason: dict[str, int] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_reason(self, reason: str, count: int) -> None:
        if count <= 0:
            return
        self.excluded_by_reason[reason] = self.excluded_by_reason.get(reason, 0) + count

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "excluded_rows": self.excluded_rows,
            "duplicates_removed": self.duplicates_removed,
            "zero_volume_excluded": self.zero_volume_excluded,
            "zero_open_interest_count": self.zero_open_interest_count,
            "missing_spot_count": self.missing_spot_count,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "source_files": list(self.source_files),
            "notes": list(self.notes),
        }


# Download (best-effort)


_NSE_BHAVCOPY_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/historical/DERIVATIVES/"
    "{year}/{mon}/fo{ddmonYYYY}bhav.csv.zip"
)

_HEADERS = {
    # NSE returns 403 or empty bodies to default python-requests User-Agent.
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/zip, application/octet-stream, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def download_bhavcopy(target_date: date, output_dir: Path) -> Path | None:
    """
    Parameters
    ----------
    target_date
        Trading date to fetch.
    output_dir
        Directory to save the downloaded file into.

    Returns
    -------
    Path or None
        Path to the downloaded file on success; ``None`` on any failure
        (network, HTTP status, empty body). Never raises on network errors.

    Notes
    -----
    NSE archives are occasionally restructured or moved behind auth. This
    function is intentionally tolerant: any failure is logged and returns
    None so the pipeline can fall back to manual import.
    """
    year = target_date.strftime("%Y")
    mon = target_date.strftime("%b").upper()
    dd = target_date.strftime("%d")
    monYYYY = target_date.strftime("%b%Y").upper()
    url = _NSE_BHAVCOPY_URL_TEMPLATE.format(year=year, mon=mon, dd=dd, ddmonYYYY=dd + monYYYY)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"fo{dd}{monYYYY}bhav.csv.zip"

    try:
        logger.info("Attempting NSE download: %s", url)
        resp = requests.get(url, headers=_HEADERS, timeout=30)
    except requests.RequestException as exc:
        logger.warning("Download failed (network): %s", exc)
        return None

    if resp.status_code != 200 or not resp.content:
        logger.warning(
            "Download failed (HTTP %s, %d bytes). NSE may be blocking or the "
            "file may not exist for this date.",
            resp.status_code, len(resp.content or b""),
        )
        return None

    try:
        out_path.write_bytes(resp.content)
    except OSError as exc:
        logger.warning("Could not write %s: %s", out_path, exc)
        return None

    logger.info("Saved %s (%d bytes)", out_path, out_path.stat().st_size)
    return out_path


# Parsing


# Column-name variants NSE has used over the years, mapped to canonical names.
_COLUMN_ALIASES: dict[str, str] = {
    # trade date
    "date": "trade_date",
    "tradedate": "trade_date",
    "trade_date": "trade_date",
    "timestamp": "trade_date",
    "business_date": "trade_date",
    "traddt": "trade_date",
    "bizdt": "trade_date",
    # symbol / underlying
    "symbol": "underlying_symbol",
    "underlying": "underlying_symbol",
    "underlying_symbol": "underlying_symbol",
    "tckrsymb": "underlying_symbol",
    # instrument type
    "instrument": "instrument_type",
    "instrument_type": "instrument_type",
    "instrumenttype": "instrument_type",
    "fininstrmtyp": "instrument_type",
    "fininstrmtp": "instrument_type",
    # expiry
    "expiry": "expiry_date",
    "expiry_dt": "expiry_date",
    "expiry_date": "expiry_date",
    "expirydate": "expiry_date",
    "xprydt": "expiry_date",
    # strike  (legacy + UDiFF)
    "strike": "strike",
    "strike_price": "strike",
    "strikeprice": "strike",
    "strikepr": "strike",
    "strike_pr": "strike",
    "strkpx": "strike",
    "strkpric": "strike",
    "strkprc": "strike",
    # option type  (legacy + UDiFF)
    "option_type": "option_type",
    "opttype": "option_type",
    "optiontype": "option_type",
    "optiontyp": "option_type",
    "option_typ": "option_type",
    "opttp": "option_type",
    "optntp": "option_type",
    # OHLC  (legacy + UDiFF)
    "open": "open", "openpric": "open", "openprc": "open", "opnpric": "open",
    "high": "high", "highpric": "high", "highprc": "high", "hghpric": "high",
    "low":  "low",  "lowpric": "low",   "lowprc":  "low",  "lwpric":  "low",
    "close": "close", "clspric": "close", "clssprc": "close",
    # settlement  (legacy + UDiFF)
    "settle_price": "settlement_price",
    "settleprice": "settlement_price",
    "settlepr": "settlement_price",
    "settle_pr": "settlement_price",
    "settlement": "settlement_price",
    "settlement_price": "settlement_price",
    "sttlmprc": "settlement_price",
    "sttlmpric": "settlement_price",
    # volume  (legacy + UDiFF)
    "contracts": "volume",
    "volume": "volume",
    "qty": "volume",
    "ttltradgvol": "volume",
    # open interest  (legacy + UDiFF)
    "open_interest": "open_interest",
    "openinterest": "open_interest",
    "openint": "open_interest",
    "open_int": "open_interest",
    "oi": "open_interest",
    "opnintrst": "open_interest",
    # underlying spot
    "underlying_spot": "underlying_spot",
    "underlying_value": "underlying_spot",
    "undrlygpric": "underlying_spot",
}

_NUMERIC_COLS = [
    "strike", "open", "high", "low", "close",
    "settlement_price", "volume", "open_interest", "underlying_spot",
]


def _read_any(path: Path) -> pd.DataFrame:
    """
    Read a raw file into a DataFrame. Handles .zip, .gz, .csv, .dat, .txt.

    The function inspects the first line to detect a delimiter. It does
    not assume a specific one.
    """
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if not names:
                raise ValueError(f"Empty zip file: {path}")
            # Read the first CSV-ish entry.
            inner_name = names[0]
            with zf.open(inner_name) as fh:
                raw = fh.read()
        df = _read_bytes(raw, source=str(path) + "!" + inner_name)
    else:
        df = _read_bytes(path.read_bytes(), source=str(path))
    return df


def _read_bytes(raw: bytes, source: str) -> pd.DataFrame:
    """Decode and parse bytes into a DataFrame with delimiter auto-detection."""
    # Try UTF-8 then latin-1.
    text: str | None = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Could not decode {source} with utf-8 or latin-1")

    first_line = text.splitlines()[0] if text else ""
    # Count candidate delimiters on the header line.
    candidates = {",": first_line.count(","), "|": first_line.count("|"),
                  ";": first_line.count(";"), "\t": first_line.count("\t")}
    sep = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    if candidates[sep] == 0:
        # Single-column header. Use comma as a harmless default; parsing will fail loudly if the file is malformed.
        sep = ","

    # NSE legacy bhavcopy has trailing whitespace-heavy columns.
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_bhavcopy(path: Path) -> pd.DataFrame:
    """
    Parse a Bhavcopy file into a DataFrame with the canonical columns.

    Parameters
    ----------
    path
        Any of .DAT, .csv, .csv.zip, .gz, .txt.

    Returns
    -------
    DataFrame
        One row per option/future contract, with canonical column names
        where the source provided them. Missing canonical columns are NaN.
    """
    df = _read_any(path)
    df = normalize_columns(df)
    df["source_file"] = str(path)
    return df


# Column normalization


def _canonical_key(name: str) -> str:
    """Lower-case, strip non-alphanumeric, for alias lookup."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


_ALIAS_LOOKUP: dict[str, str] = {_canonical_key(k): v for k, v in _COLUMN_ALIASES.items()}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map arbitrary source column names to the canonical schema.

    Unmapped source columns are dropped. Canonical columns that could not
    be matched are added as NaN so downstream code can rely on the schema.

    Additionally, instrument-type **values** are translated from NSE UDiFF
    codes (IDO, IDF, STO, STF) to the canonical instrument types
    (OPTIDX, FUTIDX, OPTSTK, FUTSTK) expected by the rest of the pipeline.
    """
    rename: dict[str, str] = {}
    for col in df.columns:
        key = _canonical_key(col)
        if key in _ALIAS_LOOKUP:
            rename[col] = _ALIAS_LOOKUP[key]

    out = df.rename(columns=rename)

    # If multiple source columns mapped to the same canonical name, keep the first non-null one and drop the rest to avoid duplicate columns.
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    # Coerce types where possible.
    for col in _NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"], errors="coerce").dt.normalize()

    # Option type cleanup.
    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].astype(str).str.strip().str.upper()

    # Underlying symbol cleanup.
    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.strip().str.upper()


    # Instrument-type value normalization
    
    if "instrument_type" in out.columns:
        out["instrument_type"] = (
            out["instrument_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace(UDIFF_INSTRUMENT_TYPE_MAP)
        )

    return out



# Cleaning


def _empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def clean_options_data(
    df: pd.DataFrame,
    cfg: Config,
    report: DataQualityReport | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Apply config-driven filters to a normalized options DataFrame.

    Every filter step records excluded counts on the report. The function
    never mutates the input in place.

    Parameters
    ----------
    df
        Output of :func:`normalize_columns` (canonical columns).
    cfg
        Loaded config.
    report
        Optional pre-existing report to append to.

    Returns
    -------
    (clean_df, report)
    """
    rep = report if report is not None else DataQualityReport()
    rep.input_rows += len(df)

    if df.empty:
        rep.notes.append("Input frame was empty.")
        return _empty_clean_frame(), rep

    work = df.copy()
    d = cfg.data

    # Instrument type filter
    before = len(work)
    work = work[work["instrument_type"].astype(str).str.upper() == d.instrument_type_filter.upper()]
    rep.add_reason("instrument_type_mismatch", before - len(work))

    # Underlying symbol filter
    before = len(work)
    work = work[work["underlying_symbol"] == d.underlying_symbol.upper()]
    rep.add_reason("underlying_symbol_mismatch", before - len(work))

    # Option type filter
    before = len(work)
    work = work[work["option_type"].isin([t.upper() for t in d.option_types])]
    rep.add_reason("option_type_mismatch", before - len(work))

    # Strike sanity
    before = len(work)
    work = work[work["strike"].notna() & (work["strike"] > 0)]
    rep.add_reason("invalid_strike", before - len(work))

    before = len(work)
    work = work[(work["strike"] >= d.min_strike) & (work["strike"] <= d.max_strike)]
    rep.add_reason("strike_out_of_range", before - len(work))

    # Date sanity
    before = len(work)
    work = work[work["trade_date"].notna() & work["expiry_date"].notna()]
    rep.add_reason("missing_date", before - len(work))

    # Time to expiry in calendar days
    tte_days = (work["expiry_date"] - work["trade_date"]).dt.days.astype(float)
    work = work.assign(_tte_days=tte_days)

    before = len(work)
    work = work[work["_tte_days"] > 0]
    rep.add_reason("expiry_on_or_before_trade_date", before - len(work))

    before = len(work)
    work = work[(work["_tte_days"] >= d.min_time_to_expiry_days)
                & (work["_tte_days"] <= d.max_time_to_expiry_days)]
    rep.add_reason("time_to_expiry_out_of_range", before - len(work))

    # Price sanity: require at least one usable price field
    before = len(work)
    usable_price = (
        work["settlement_price"].notna() & (work["settlement_price"] > 0)
    ) | (
        work["close"].notna() & (work["close"] > 0)
    )
    work = work[usable_price]
    rep.add_reason("no_usable_price", before - len(work))

    # Duplicate contract removal
    if d.exclude_duplicate_contracts:
        key_cols = ["trade_date", "expiry_date", "strike", "option_type"]
        before = len(work)
        work = work.drop_duplicates(subset=key_cols, keep="first")
        removed = before - len(work)
        rep.duplicates_removed += removed
        rep.add_reason("duplicate_contract", removed)

    # Zero open interest diagnostic (does NOT drop rows)
    if "open_interest" in work.columns:
        rep.zero_open_interest_count += int((work["open_interest"].fillna(0) == 0).sum())

    # Zero volume (primary traded sample)
    if d.exclude_zero_volume:
        before = len(work)
        work = work[work["volume"].fillna(0) >= d.min_volume]
        removed = before - len(work)
        rep.zero_volume_excluded += removed
        rep.add_reason("zero_or_low_volume", removed)

    # Underlying spot missing (diagnostic only)
    if work["underlying_spot"].isna().any():
        n_missing = int(work["underlying_spot"].isna().sum())
        rep.missing_spot_count += n_missing
        rep.notes.append(
            f"{n_missing} rows have no underlying_spot; these will be dropped "
            f"before pricing because model inputs require a spot."
        )

    # Final drop of rows still missing spot; we cannot price without S.
    before = len(work)
    work = work[work["underlying_spot"].notna() & (work["underlying_spot"] > 0)]
    rep.add_reason("missing_or_invalid_spot", before - len(work))

    work = work.drop(columns=["_tte_days"], errors="ignore")
    rep.output_rows += len(work)
    rep.excluded_rows = rep.input_rows - rep.output_rows

    return work.reset_index(drop=True), rep



# Spot matching


def load_spot_prices(spot_file: Path) -> pd.DataFrame:
    """
    Load a spot-price CSV with columns ``trade_date`` and ``underlying_spot``.

    Accepts column-name variants. Returns a two-column DataFrame.
    """
    if not spot_file.is_file():
        raise FileNotFoundError(f"Spot file not found: {spot_file}")
    df = pd.read_csv(spot_file)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {_canonical_key(c): c for c in df.columns}
    date_col = cols.get("tradedate") or cols.get("date")
    spot_col = cols.get("underlyingspot") or cols.get("close") or cols.get("spot")
    if date_col is None or spot_col is None:
        raise ValueError(
            f"Spot file must have a date column and a spot column. "
            f"Found columns: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "trade_date": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "underlying_spot": pd.to_numeric(df[spot_col], errors="coerce"),
    })
    out = out.dropna().drop_duplicates(subset=["trade_date"])
    return out


def _derive_spot_from_futures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback spot proxy: use the near-month NIFTY futures settlement price.

    This is only used when no explicit spot file is supplied. The result is
    an approximation and is flagged in the report. Returns a DataFrame
    with trade_date and underlying_spot.

    Both the canonical futures types (FUTIDX, FUTURES) and the
    raw UDiFF codes (IDF, STF) are accepted, so this works even
    if :func:`normalize_columns` has not yet been applied.
    """
    fut_types = {"FUTIDX", "FUTURES", "IDF", "STF"}
    fut = df[df["instrument_type"].astype(str).str.upper().isin(fut_types)].copy()
    if fut.empty:
        return pd.DataFrame(columns=["trade_date", "underlying_spot"])
    # Take the nearest expiry per trade date.
    fut = fut.sort_values(["trade_date", "expiry_date"])
    fut = fut.drop_duplicates(subset=["trade_date"], keep="first")
    price_col = "settlement_price" if fut["settlement_price"].notna().any() else "close"
    out = fut[["trade_date", price_col]].rename(columns={price_col: "underlying_spot"})
    return out.dropna()


def attach_spot(
    options: pd.DataFrame,
    cfg: Config,
    report: DataQualityReport,
) -> pd.DataFrame:
    """
    Attach underlying_spot to options by trade date.

    Order of preference:
    1. data/raw/spot.csv (or data/raw/nifty_spot.csv) if present.
    2. A underlying_spot column already on the options frame.
    3. Near-month NIFTY futures settlement as a proxy (flagged).
    """
    work = options.copy()

    # Explicit spot file
    candidates = [
        cfg.paths.data_raw / "spot.csv",
        cfg.paths.data_raw / "nifty_spot.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            rep = load_spot_prices(cand)
            work = work.drop(columns=["underlying_spot"], errors="ignore")
            work = work.merge(rep, on="trade_date", how="left")
            report.notes.append(f"Spot prices loaded from {cand.name}.")
            return work

    # Column already present in options frame
    if "underlying_spot" in work.columns and work["underlying_spot"].notna().any():
        report.notes.append("Underlying spot column found in the raw file.")
        return work

    # Futures proxy
    proxy = _derive_spot_from_futures(work)
    if not proxy.empty:
        work = work.drop(columns=["underlying_spot"], errors="ignore")
        work = work.merge(proxy, on="trade_date", how="left")
        report.notes.append(
            "No explicit spot file found. Underlying spot was approximated "
            "using the near-month NIFTY futures settlement price. This is "
            "an approximation, not the cash index."
        )
        return work

    report.notes.append(
        "No spot source available. Pricing requires spot; rows will be dropped."
    )
    return work


# Top-level loader


def load_all_raw(
    cfg: Config,
    report: DataQualityReport | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Load every Bhavcopy file in cfg.paths.data_raw and concatenate.

    Files are expected to be NSE F&O Bhavcopy in .DAT, .csv, .csv.zip, or .gz
    form. Unrecognized files are skipped with a logged warning.
    """
    rep = report if report is not None else DataQualityReport()
    raw_dir = cfg.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {".dat", ".csv", ".zip", ".gz", ".txt"}
    files = [p for p in sorted(raw_dir.iterdir())
             if p.is_file() and p.suffix.lower() in suffixes]

    if not files:
        rep.notes.append(
            f"No Bhavcopy files found in {raw_dir}. Place NSE files there, "
            f"or run the download script."
        )
        return _empty_clean_frame(), rep

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = parse_bhavcopy(f)
            frames.append(df)
            rep.source_files.append(f.name)
            logger.info("Parsed %s -> %d rows", f.name, len(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", f.name, exc)
            rep.notes.append(f"Failed to parse {f.name}: {exc}")

    if not frames:
        return _empty_clean_frame(), rep

    combined = pd.concat(frames, ignore_index=True)
    combined = attach_spot(combined, cfg, rep)

    cleaned, rep = clean_options_data(combined, cfg, rep)
    return cleaned, rep