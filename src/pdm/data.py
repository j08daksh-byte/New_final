"""Loading and labelling of the NASA C-MAPSS FD001 run-to-failure data."""
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", "op1", "op2", "op3", *SENSORS]
RUL_CAP = 125  # engines are treated as healthy above this; standard piecewise-linear target


def load_cycles(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_test_rul(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None)[0].to_numpy(dtype=float)


def add_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Each training engine runs to failure, so RUL = last cycle - current cycle (capped)."""
    out = train.copy()
    out["rul"] = (out.groupby("unit")["cycle"].transform("max") - out["cycle"]).clip(upper=cap)
    return out


def informative_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors that are constant or two-valued in FD001 carry no degradation signal."""
    return [s for s in SENSORS if train[s].nunique() > 2]
