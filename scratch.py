import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import pandas as pd

ROOT = Path("d:/Webs/249/Turbofan-Predictive-Maintenance")
sys.path.insert(0, str(ROOT / "src"))

from pdm.data import RUL_CAP, add_rul, informative_sensors, load_cycles, load_test_rul
from pdm.features import WINDOW, build_features, last_cycle_rows
from pdm.model import gradient_boosting, clip_rul, SEED

def main():
    train = add_rul(load_cycles(ROOT / "data/train_FD001.txt"))
    test = load_cycles(ROOT / "data/test_FD001.txt")
    true_rul = load_test_rul(ROOT / "data/RUL_FD001.txt")
    sensors = informative_sensors(train)

    X, y, groups = build_features(train, sensors), train["rul"], train["unit"]
    X_test_all = build_features(test, sensors)
    last = last_cycle_rows(test)
    X_test = X_test_all.loc[last]
    true_capped = np.minimum(true_rul, RUL_CAP)

    # Standard conformal margin (all rows)
    np.random.seed(SEED)
    scores_all = []
    
    # Subsampled conformal margin (1 random row per engine)
    scores_sub = []
    
    lo_preds = []
    hi_preds = []

    model = gradient_boosting().fit(X, y)
    lo_m = gradient_boosting("quantile", 0.1).fit(X, y)
    hi_m = gradient_boosting("quantile", 0.9).fit(X, y)
    
    pred = clip_rul(model.predict(X_test))
    test_lo = clip_rul(lo_m.predict(X_test))
    test_hi = clip_rul(hi_m.predict(X_test))
    
    # Standard absolute residual conformal prediction
    scores_res = np.empty(len(y))
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = gradient_boosting().fit(X.iloc[tr], y.iloc[tr])
        pred_va = m.predict(X.iloc[va])
        scores_res[va] = np.abs(pred_va - y.iloc[va].values)
    
    margin_res = float(np.quantile(scores_res, 0.8))
    
    print(f"Margin (Residuals): {margin_res:.2f}")
    
    cov_res = np.mean((true_capped >= clip_rul(pred - margin_res)) & (true_capped <= clip_rul(pred + margin_res)))
    
    print(f"Test Coverage (Residuals): {cov_res:.3f}")

if __name__ == '__main__':
    main()
