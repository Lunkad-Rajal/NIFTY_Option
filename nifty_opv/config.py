"""
Loads ``config.yaml`` into typed dataclasses.
Do not parse YAML anywhere else, and do
not hardcode values that belong in the config file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Nested dataclasses mirroring config.yaml


@dataclass(frozen=True)
class ProjectCfg:
    name: str
    version: str
    random_seed: int


@dataclass(frozen=True)
class PathsCfg:
    data_raw: Path
    data_processed: Path
    reports_figures: Path
    reports_tables: Path
    cache: Path


@dataclass(frozen=True)
class DataCfg:
    source: str
    instrument_type_filter: str
    underlying_symbol: str
    option_types: tuple[str, ...]
    expiry_preference: str
    date_start: str
    date_end: str
    manual_files: tuple[str, ...]
    min_strike: float
    max_strike: float
    min_volume: int
    min_open_interest: int
    min_time_to_expiry_days: float
    max_time_to_expiry_days: float
    exclude_zero_volume: bool
    exclude_duplicate_contracts: bool
    spot_matching: str


@dataclass(frozen=True)
class MarketCfg:
    risk_free_rate: float
    risk_free_rate_source: str
    dividend_yield: float
    dividend_yield_source: str
    day_count: str
    exercise_style: str
    settlement: str


@dataclass(frozen=True)
class BlackScholesCfg:
    price_field: str
    min_T: float
    min_sigma: float


@dataclass(frozen=True)
class CRRCfg:
    default_steps: int
    convergence_steps: tuple[int, ...]
    american: bool


@dataclass(frozen=True)
class ModelsCfg:
    black_scholes: BlackScholesCfg
    crr: CRRCfg


@dataclass(frozen=True)
class ImpliedVolCfg:
    method: str
    lo: float
    hi: float
    tol: float
    max_bracket_expansions: int
    bracket_multiplier: float
    repricing_tol: float


@dataclass(frozen=True)
class GreeksTolerances:
    delta: float
    gamma: float
    vega: float
    rho: float
    theta: float


@dataclass(frozen=True)
class GreeksBumps:
    spot: float
    vol: float
    rate: float
    time: float


@dataclass(frozen=True)
class GreeksCfg:
    fd_bumps: GreeksBumps
    tolerances: GreeksTolerances


@dataclass(frozen=True)
class IVRepricingCfg:
    enabled: bool


@dataclass(frozen=True)
class ConstantATMCfg:
    enabled: bool
    atm_rule: str
    moneyness_metric: str


@dataclass(frozen=True)
class CRRExperimentCfg:
    enabled: bool
    steps: int


@dataclass(frozen=True)
class ExperimentsCfg:
    iv_repricing: IVRepricingCfg
    constant_atm_vol: ConstantATMCfg
    crr_tree: CRRExperimentCfg


@dataclass(frozen=True)
class MoneynessBucketsCfg:
    edges: tuple[float, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class PlotsCfg:
    dpi: int
    figsize: tuple[int, int]
    style: str
    palette: str
    save_format: str


@dataclass(frozen=True)
class LoggingCfg:
    level: str
    format: str


@dataclass(frozen=True)
class Config:
    project: ProjectCfg
    paths: PathsCfg
    data: DataCfg
    market: MarketCfg
    models: ModelsCfg
    implied_vol: ImpliedVolCfg
    greeks: GreeksCfg
    experiments: ExperimentsCfg
    moneyness_buckets: MoneynessBucketsCfg
    plots: PlotsCfg
    logging: LoggingCfg
    project_root: Path = field(default=Path("."))


# Loading


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing config key '{key}' under '{where}'")
    return d[key]


def load_config(path: str | Path = "config.yaml") -> Config:
    """
    Load a YAML config file into a typed :class:`Config`.

    Parameters
    ----------
    path
        Path to the YAML file. Defaults to ``config.yaml`` in the current
        working directory.

    Returns
    -------
    Config
        Fully populated, frozen dataclass.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    KeyError
        If a required key is missing.
    """
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise TypeError(f"Config root must be a mapping, got {type(raw).__name__}")

    project_root = cfg_path.parent

    project = ProjectCfg(
        name=_require(_require(raw, "project", "root"), "name", "project"),
        version=_require(raw["project"], "version", "project"),
        random_seed=int(_require(raw["project"], "random_seed", "project")),
    )

    p = _require(raw, "paths", "root")
    paths = PathsCfg(
        data_raw=project_root / _require(p, "data_raw", "paths"),
        data_processed=project_root / _require(p, "data_processed", "paths"),
        reports_figures=project_root / _require(p, "reports_figures", "paths"),
        reports_tables=project_root / _require(p, "reports_tables", "paths"),
        cache=project_root / _require(p, "cache", "paths"),
    )

    d = _require(raw, "data", "root")
    data = DataCfg(
        source=str(_require(d, "source", "data")),
        instrument_type_filter=str(_require(d, "instrument_type_filter", "data")),
        underlying_symbol=str(_require(d, "underlying_symbol", "data")),
        option_types=tuple(_require(d, "option_types", "data")),
        expiry_preference=str(_require(d, "expiry_preference", "data")),
        date_start=str(_require(d, "date_start", "data")),
        date_end=str(_require(d, "date_end", "data")),
        manual_files=tuple(_require(d, "manual_files", "data") or ()),
        min_strike=float(_require(d, "min_strike", "data")),
        max_strike=float(_require(d, "max_strike", "data")),
        min_volume=int(_require(d, "min_volume", "data")),
        min_open_interest=int(_require(d, "min_open_interest", "data")),
        min_time_to_expiry_days=float(_require(d, "min_time_to_expiry_days", "data")),
        max_time_to_expiry_days=float(_require(d, "max_time_to_expiry_days", "data")),
        exclude_zero_volume=bool(_require(d, "exclude_zero_volume", "data")),
        exclude_duplicate_contracts=bool(_require(d, "exclude_duplicate_contracts", "data")),
        spot_matching=str(_require(d, "spot_matching", "data")),
    )

    m = _require(raw, "market", "root")
    market = MarketCfg(
        risk_free_rate=float(_require(m, "risk_free_rate", "market")),
        risk_free_rate_source=str(_require(m, "risk_free_rate_source", "market")),
        dividend_yield=float(_require(m, "dividend_yield", "market")),
        dividend_yield_source=str(_require(m, "dividend_yield_source", "market")),
        day_count=str(_require(m, "day_count", "market")),
        exercise_style=str(_require(m, "exercise_style", "market")),
        settlement=str(_require(m, "settlement", "market")),
    )

    models_raw = _require(raw, "models", "root")
    bs_raw = _require(models_raw, "black_scholes", "models")
    crr_raw = _require(models_raw, "crr", "models")
    models = ModelsCfg(
        black_scholes=BlackScholesCfg(
            price_field=str(_require(bs_raw, "price_field", "models.black_scholes")),
            min_T=float(_require(bs_raw, "min_T", "models.black_scholes")),
            min_sigma=float(_require(bs_raw, "min_sigma", "models.black_scholes")),
        ),
        crr=CRRCfg(
            default_steps=int(_require(crr_raw, "default_steps", "models.crr")),
            convergence_steps=tuple(int(x) for x in _require(
                crr_raw, "convergence_steps", "models.crr")),
            american=bool(_require(crr_raw, "american", "models.crr")),
        ),
    )

    iv = _require(raw, "implied_vol", "root")
    implied_vol = ImpliedVolCfg(
        method=str(_require(iv, "method", "implied_vol")),
        lo=float(_require(iv, "lo", "implied_vol")),
        hi=float(_require(iv, "hi", "implied_vol")),
        tol=float(_require(iv, "tol", "implied_vol")),
        max_bracket_expansions=int(_require(iv, "max_bracket_expansions", "implied_vol")),
        bracket_multiplier=float(_require(iv, "bracket_multiplier", "implied_vol")),
        repricing_tol=float(_require(iv, "repricing_tol", "implied_vol")),
    )

    g = _require(raw, "greeks", "root")
    bumps_raw = _require(g, "fd_bumps", "greeks")
    tols_raw = _require(g, "tolerances", "greeks")
    greeks = GreeksCfg(
        fd_bumps=GreeksBumps(
            spot=float(_require(bumps_raw, "spot", "greeks.fd_bumps")),
            vol=float(_require(bumps_raw, "vol", "greeks.fd_bumps")),
            rate=float(_require(bumps_raw, "rate", "greeks.fd_bumps")),
            time=float(_require(bumps_raw, "time", "greeks.fd_bumps")),
        ),
        tolerances=GreeksTolerances(
            delta=float(_require(tols_raw, "delta", "greeks.tolerances")),
            gamma=float(_require(tols_raw, "gamma", "greeks.tolerances")),
            vega=float(_require(tols_raw, "vega", "greeks.tolerances")),
            rho=float(_require(tols_raw, "rho", "greeks.tolerances")),
            theta=float(_require(tols_raw, "theta", "greeks.tolerances")),
        ),
    )

    e = _require(raw, "experiments", "root")
    ivr = _require(e, "iv_repricing", "experiments")
    cav = _require(e, "constant_atm_vol", "experiments")
    ct = _require(e, "crr_tree", "experiments")
    experiments = ExperimentsCfg(
        iv_repricing=IVRepricingCfg(enabled=bool(_require(ivr, "enabled", "experiments.iv_repricing"))),
        constant_atm_vol=ConstantATMCfg(
            enabled=bool(_require(cav, "enabled", "experiments.constant_atm_vol")),
            atm_rule=str(_require(cav, "atm_rule", "experiments.constant_atm_vol")),
            moneyness_metric=str(_require(cav, "moneyness_metric", "experiments.constant_atm_vol")),
        ),
        crr_tree=CRRExperimentCfg(
            enabled=bool(_require(ct, "enabled", "experiments.crr_tree")),
            steps=int(_require(ct, "steps", "experiments.crr_tree")),
        ),
    )

    mb = _require(raw, "moneyness_buckets", "root")
    moneyness_buckets = MoneynessBucketsCfg(
        edges=tuple(float(x) for x in _require(mb, "edges", "moneyness_buckets")),
        labels=tuple(str(x) for x in _require(mb, "labels", "moneyness_buckets")),
    )

    p2 = _require(raw, "plots", "root")
    plots = PlotsCfg(
        dpi=int(_require(p2, "dpi", "plots")),
        figsize=tuple(int(x) for x in _require(p2, "figsize", "plots")),
        style=str(_require(p2, "style", "plots")),
        palette=str(_require(p2, "palette", "plots")),
        save_format=str(_require(p2, "save_format", "plots")),
    )

    lg = _require(raw, "logging", "root")
    logging_cfg = LoggingCfg(
        level=str(_require(lg, "level", "logging")),
        format=str(_require(lg, "format", "logging")),
    )

    cfg = Config(
        project=project,
        paths=paths,
        data=data,
        market=market,
        models=models,
        implied_vol=implied_vol,
        greeks=greeks,
        experiments=experiments,
        moneyness_buckets=moneyness_buckets,
        plots=plots,
        logging=logging_cfg,
        project_root=project_root,
    )

    _validate(cfg)
    _ensure_dirs(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Cross-field sanity checks that individual field parsing cannot do."""
    if cfg.data.min_strike >= cfg.data.max_strike:
        raise ValueError(
            f"data.min_strike ({cfg.data.min_strike}) must be < "
            f"data.max_strike ({cfg.data.max_strike})"
        )
    if cfg.data.min_time_to_expiry_days >= cfg.data.max_time_to_expiry_days:
        raise ValueError(
            "data.min_time_to_expiry_days must be < data.max_time_to_expiry_days"
        )
    if cfg.implied_vol.lo <= 0.0 or cfg.implied_vol.hi <= cfg.implied_vol.lo:
        raise ValueError(
            f"implied_vol bracket invalid: lo={cfg.implied_vol.lo}, hi={cfg.implied_vol.hi}"
        )
    if len(cfg.moneyness_buckets.edges) != len(cfg.moneyness_buckets.labels) + 1:
        raise ValueError(
            "moneyness_buckets: len(edges) must equal len(labels) + 1, got "
            f"{len(cfg.moneyness_buckets.edges)} edges and "
            f"{len(cfg.moneyness_buckets.labels)} labels"
        )
    if cfg.market.exercise_style.lower() not in {"european", "american"}:
        raise ValueError(
            f"market.exercise_style must be 'european' or 'american', got "
            f"{cfg.market.exercise_style!r}"
        )
    if cfg.market.day_count not in {"calendar_days_365", "trading_days_252"}:
        raise ValueError(
            f"market.day_count must be 'calendar_days_365' or 'trading_days_252', got "
            f"{cfg.market.day_count!r}"
        )


def _ensure_dirs(cfg: Config) -> None:
    """Create output directories if they do not exist."""
    for p in (
        cfg.paths.data_raw,
        cfg.paths.data_processed,
        cfg.paths.reports_figures,
        cfg.paths.reports_tables,
        cfg.paths.cache,
    ):
        p.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config) -> None:
    """Configure the root logger from the config."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format=cfg.logging.format,
    )


__all__ = [
    "Config",
    "load_config",
    "setup_logging",
]