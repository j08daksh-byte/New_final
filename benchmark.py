"""Robustness Benchmark across C-MAPSS subsets."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from pdm.data import RUL_CAP, add_rul, informative_sensors, load_cycles, load_test_rul
from pdm.features import build_features, fit_baselines, last_cycle_rows
from pdm.model import clip_rul, gradient_boosting, nasa_score, rmse, alert_quality

def evaluate_subset(train_name, test_name):
    # Load Train
    train_df = add_rul(load_cycles(ROOT / f"data/train_{train_name}.txt"))
    sensors = informative_sensors(train_df)
    
    # In FD002/004, operating conditions vary. If we don't include them, 
    # we're blindly assuming FD001 preprocessing (sensor only).
    baselines = fit_baselines(train_df, sensors)
    
    X_train = build_features(train_df, sensors, baselines=baselines)
    y_train = train_df["rul"]
    train_engines = int(train_df["unit"].nunique())
    
    # Load Test
    test_df = load_cycles(ROOT / f"data/test_{test_name}.txt")
    true_rul = load_test_rul(ROOT / f"data/RUL_{test_name}.txt")
    test_engines = int(test_df["unit"].nunique())
    
    X_test_all = build_features(test_df, sensors, baselines=baselines)
    last = last_cycle_rows(test_df)
    X_test = X_test_all.loc[last]
    true_capped = np.minimum(true_rul, RUL_CAP)
    
    # Train Point Model
    model = gradient_boosting().fit(X_train, y_train)
    pred = clip_rul(model.predict(X_test))
    
    # Train Uncertainty Models
    lo_m = gradient_boosting("quantile", 0.1).fit(X_train, y_train)
    hi_m = gradient_boosting("quantile", 0.9).fit(X_train, y_train)
    
    lo = clip_rul(lo_m.predict(X_test))
    hi = clip_rul(hi_m.predict(X_test))
    
    test_rmse = rmse(true_capped, pred)
    n_score = nasa_score(true_capped, pred)
    coverage = float(np.mean((true_capped >= lo) & (true_capped <= hi)))
    mean_width = float(np.mean(hi - lo))
    
    alert = alert_quality(true_capped, pred, 30)
    
    return {
        "Train": train_name,
        "Test": test_name,
        "Train N": train_engines,
        "Test N": test_engines,
        "RMSE": round(test_rmse, 2),
        "NASA Score": round(n_score, 1),
        "Coverage": round(coverage, 3),
        "Width": round(mean_width, 1),
        "Alert Rec": round(alert["recall"] if alert["recall"] is not None else 0.0, 2),
        "Alert Prec": round(alert["precision"] if alert["precision"] is not None else 0.0, 2)
    }

def main():
    results = []
    # 1. Same-dataset benchmarks
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"Evaluating {ds}...")
        try:
            results.append(evaluate_subset(ds, ds))
        except Exception as e:
            print(f"Failed {ds}: {e}")
            
    # 2. Cross-dataset experiments
    # Train on FD001 (1 op, 1 fault), Test on FD003 (1 op, 2 faults)
    print("Evaluating Cross-Dataset FD001 -> FD003...")
    try:
        results.append(evaluate_subset("FD001", "FD003"))
    except Exception as e:
        print(f"Failed FD001 -> FD003: {e}")
        
    # Train on FD002 (6 op, 1 fault), Test on FD004 (6 op, 2 faults)
    print("Evaluating Cross-Dataset FD002 -> FD004...")
    try:
        results.append(evaluate_subset("FD002", "FD004"))
    except Exception as e:
        print(f"Failed FD002 -> FD004: {e}")
        
    print("\n--- ROBUSTNESS BENCHMARK RESULTS ---")
    
    # Manual markdown formatting to avoid tabulate dependency
    headers = ["Train", "Test", "Train N", "Test N", "RMSE", "NASA Score", "Coverage", "Width", "Alert Rec", "Alert Prec"]
    md = f"| {' | '.join(headers)} |\n| {' | '.join(['---'] * len(headers))} |\n"
    for r in results:
        md += f"| {r.get('Train','')} | {r.get('Test','')} | {r.get('Train N','')} | {r.get('Test N','')} | {r.get('RMSE','')} | {r.get('NASA Score','')} | {r.get('Coverage','')} | {r.get('Width','')} | {r.get('Alert Rec','')} | {r.get('Alert Prec','')} |\n"
    
    print(md)
    
    # Save the markdown directly for the dashboard integration
    with open("ROBUSTNESS_RESULTS.md", "w") as f:
        f.write(md)
    
if __name__ == "__main__":
    main()
