"""
Configurable data filters (YAML: data.filters).

    data:
      filters:
        - name: leagues                 # registered filter
          values: [E0, SP1, D1, I1, F1] # leagues to KEEP
          stage: split                  # 'split' (default) | 'load'

Stages:
    - load : right after reading the raw data (workflow.run), before any feature is computed.
    - split: at the end of Dataset.split_data(), on train/val/test (features already computed,
             so e.g. promoted teams keep their lower-division history in lags/ratings).

Backward compatibility: the legacy key ``data.leagues`` is a shortcut. ``null``, ``[]`` or
``[null]`` mean "all leagues" (no filter); otherwise it becomes ``{name: leagues, values, stage: split}``.

To add a filter: write ``fn(df, cfg) -> df`` and register it in FILTERS.
"""
from typing import Callable, Dict, List
import pandas as pd

STAGES = ("load", "split")
DEFAULT_STAGE = "split"


def _log(msg: str):
    print(f"INFO - {msg}")


def filter_leagues(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Keep only the rows whose league (Div) is in cfg['values']."""
    if "Div" not in df.columns:
        raise ValueError("Filter 'leagues' needs a 'Div' column, which is not in the data")
    keep = list(cfg["values"])
    missing = sorted(set(keep) - set(df["Div"].dropna().unique()))
    if missing:
        _log(f"WARNING filtro leagues: ligas del YAML que no aparecen en los datos: {missing}")
    return df[df["Div"].isin(keep)]


FILTERS: Dict[str, Callable[[pd.DataFrame, Dict], pd.DataFrame]] = {
    "leagues": filter_leagues,
}


def _is_all(values) -> bool:
    """None, [] or [None] -> no restriction."""
    if values is None:
        return True
    if isinstance(values, (list, tuple)):
        return len([v for v in values if v is not None]) == 0
    return False


def normalize_filters(data_cfg: Dict) -> List[Dict]:
    """Return the validated list of filters (legacy shortcut applied, default stage set).
    Raises ValueError on an invalid configuration."""
    data_cfg = data_cfg or {}
    raw = data_cfg.get("filters") or []
    if not isinstance(raw, list):
        raise ValueError("data.filters must be a list of {name, values, stage}")
    filters = []
    for i, f in enumerate(raw):
        if not isinstance(f, dict) or "name" not in f:
            raise ValueError(f"data.filters[{i}] must be a dict with at least 'name'")
        name = f["name"]
        if name not in FILTERS:
            raise ValueError(f"data.filters[{i}]: unknown filter '{name}'. Available: {sorted(FILTERS)}")
        stage = f.get("stage", DEFAULT_STAGE)
        if stage not in STAGES:
            raise ValueError(f"data.filters[{i}] ({name}): invalid stage '{stage}'. Allowed: {list(STAGES)}")
        values = f.get("values")
        if _is_all(values) or not isinstance(values, (list, tuple)):
            raise ValueError(f"data.filters[{i}] ({name}): 'values' must be a non-empty list")
        filters.append({**f, "name": name, "stage": stage, "values": [v for v in values if v is not None]})

    legacy = data_cfg.get("leagues")
    if not _is_all(legacy):
        if not isinstance(legacy, (list, tuple)):
            raise ValueError("data.leagues must be a list of league codes (or null)")
        if any(f["name"] == "leagues" for f in filters):
            raise ValueError("Ambiguous config: both data.leagues and a 'leagues' entry in data.filters are set")
        filters.append({"name": "leagues", "values": [v for v in legacy if v is not None], "stage": DEFAULT_STAGE})
    return filters


def apply_filters(df: pd.DataFrame, data_cfg: Dict, stage: str, label: str = "") -> pd.DataFrame:
    """Apply, in order, the filters configured for ``stage``."""
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'")
    for f in normalize_filters(data_cfg):
        if f["stage"] != stage:
            continue
        n0 = len(df)
        df = FILTERS[f["name"]](df, f)
        _log(f"filtro {f['name']} [{stage}{'/' + label if label else ''}]: {n0} -> {len(df)} filas")
    return df
