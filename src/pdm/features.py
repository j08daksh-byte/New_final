"""Per-engine features. Everything is causal: a row only uses cycles up to and including itself."""
import pandas as pd

WINDOW = 20
BASELINE_CYCLES = 10


def build_features(df: pd.DataFrame, sensors: list[str], window: int = WINDOW) -> pd.DataFrame:
    grouped = df.groupby("unit", sort=False)
    parts = {"cycle": df["cycle"]}
    for s in sensors:
        roll = grouped[s].rolling(window, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        parts[s] = df[s]
        parts[f"{s}_mean"] = mean
        parts[f"{s}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        # Drift from the engine's own early-life level removes unit-to-unit manufacturing offsets.
        healthy = grouped[s].transform(lambda x: x.iloc[:BASELINE_CYCLES].mean())
        parts[f"{s}_drift"] = mean - healthy
    return pd.DataFrame(parts, index=df.index)


def last_cycle_rows(df: pd.DataFrame) -> pd.Index:
    return df.groupby("unit")["cycle"].idxmax()
