"""Per-engine features. Everything is causal: a row only uses cycles up to and including itself."""
import pandas as pd

WINDOW = 20
BASELINE_CYCLES = 10
    
def fit_baselines(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Calculate global healthy baseline for each operating condition regime to decouple altitude noise from degradation."""
    if "op1" not in df.columns:
        return None
    healthy = df[df["cycle"] <= BASELINE_CYCLES].copy()
    healthy["regime"] = healthy["op1"].round(0)
    return healthy.groupby("regime")[sensors].mean()


def build_features(df: pd.DataFrame, sensors: list[str], window: int = WINDOW, baselines: pd.DataFrame = None) -> pd.DataFrame:
    parts = {"cycle": df["cycle"]}
    
    df_norm = pd.DataFrame(index=df.index)
    if baselines is not None and "op1" in df.columns:
        regime = df["op1"].round(0)
        for s in sensors:
            b = regime.map(baselines[s]).fillna(0)
            df_norm[s] = df[s] - b
    else:
        for s in sensors:
            df_norm[s] = df[s]
            
    df_norm["unit"] = df["unit"]
    grouped = df_norm.groupby("unit", sort=False)
            
    for s in sensors:
        # Group by unit, but operate on the NORMALIZED sensor
        roll = grouped[s].rolling(window, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        parts[s] = df_norm[s]
        parts[f"{s}_mean"] = mean
        parts[f"{s}_std"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        # Drift from the engine's own early-life level removes unit-to-unit manufacturing offsets.
        healthy = grouped[s].transform(lambda x: x.iloc[:BASELINE_CYCLES].mean())
        parts[f"{s}_drift"] = mean - healthy
    return pd.DataFrame(parts, index=df.index)


def last_cycle_rows(df: pd.DataFrame) -> pd.Index:
    return df.groupby("unit")["cycle"].idxmax()
