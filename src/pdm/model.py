"""Models, metrics and the maintenance decision rule."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdm.data import RUL_CAP

SEED = 42
# Project-defined bands (cycles of remaining life). They are assumptions, not an airline standard.
BANDS = [("CRITICAL", 20), ("HIGH", 50), ("MEDIUM", 80)]


def gradient_boosting(loss: str = "squared_error", quantile: float | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED,
    )


def ridge() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def clip_rul(pred: np.ndarray) -> np.ndarray:
    return np.clip(pred, 0, RUL_CAP)


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nasa_score(true: np.ndarray, pred: np.ndarray) -> float:
    """PHM08 score: late predictions (overestimating life) are penalised more than early ones."""
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def band(rul: float) -> str:
    for name, limit in BANDS:
        if rul <= limit:
            return name
    return "LOW"


def alert_quality(true: np.ndarray, pred: np.ndarray, threshold: float) -> dict:
    """Treat 'predicted RUL <= threshold' as a maintenance alert for engines truly within it."""
    true, pred = np.asarray(true), np.asarray(pred)
    actual, flagged = true <= threshold, pred <= threshold
    tp = int(np.sum(actual & flagged))
    return {
        "threshold": threshold, "due": int(actual.sum()), "flagged": int(flagged.sum()), "caught": tp,
        "recall": round(tp / actual.sum(), 3) if actual.sum() else None,
        "precision": round(tp / flagged.sum(), 3) if flagged.sum() else None,
    }
