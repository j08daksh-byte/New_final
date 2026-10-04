"""Train, evaluate and export everything the dashboard needs.

    python run.py            # writes docs/data.json and prints the result table
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from pdm.data import RUL_CAP, add_rul, informative_sensors, load_cycles, load_test_rul  # noqa: E402
from pdm.features import WINDOW, build_features, fit_baselines, last_cycle_rows  # noqa: E402
from pdm.model import (  # noqa: E402
    BANDS, SEED, alert_quality, band, clip_rul, gradient_boosting, nasa_score, ridge, rmse,
    uncertainty_status, risk_category, calibrate_margin
)
from pdm.simulator import simulate_fleet  # noqa: E402
from pdm.graph import build_demo_fleet_graph  # noqa: E402

TREND_SENSORS = ["s11", "s4", "s12", "s7"]
SENSOR_NAMES = {
    "s2": "LPC outlet temperature", "s3": "HPC outlet temperature", "s4": "LPT outlet temperature",
    "s7": "HPC outlet pressure", "s8": "Physical fan speed", "s9": "Physical core speed",
    "s11": "HPC outlet static pressure", "s12": "Fuel flow / Ps30 ratio", "s13": "Corrected fan speed",
    "s14": "Corrected core speed", "s15": "Bypass ratio", "s17": "Bleed enthalpy",
    "s20": "HPT coolant bleed", "s21": "LPT coolant bleed",
}


def cross_validate(make, X, y, groups) -> float:
    errs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        errs.append(rmse(y.iloc[va], clip_rul(m.predict(X.iloc[va]))))
    return float(np.mean(errs))

def main() -> None:
    train = add_rul(load_cycles(ROOT / "data/train_FD001.txt"))
    test = load_cycles(ROOT / "data/test_FD001.txt")
    true_rul = load_test_rul(ROOT / "data/RUL_FD001.txt")
    sensors = informative_sensors(train)
    baselines = fit_baselines(train, sensors)

    X, y, groups = build_features(train, sensors, baselines=baselines), train["rul"], train["unit"]
    X_test_all = build_features(test, sensors, baselines=baselines)
    last = last_cycle_rows(test)
    X_test, raw_cols = X_test_all.loc[last], ["cycle", *sensors]
    true_capped = np.minimum(true_rul, RUL_CAP)

    candidates = {
        "Constant (training median)": None,
        "Ridge regression, raw sensors": (ridge, raw_cols),
        "Gradient boosting, raw sensors": (gradient_boosting, raw_cols),
        "Gradient boosting, rolling + drift features": (gradient_boosting, list(X.columns)),
    }
    rows, preds = [], {}
    for name, spec in candidates.items():
        if spec is None:
            p = np.full(len(true_rul), float(y.median()))
            cv = rmse(y, np.full(len(y), float(y.median())))
        else:
            make, cols = spec
            cv = cross_validate(make, X[cols], y, groups)
            p = clip_rul(make().fit(X[cols], y).predict(X_test[cols]))
        preds[name] = p
        rows.append({
            "model": name, "cv_rmse": round(cv, 2), "test_rmse": round(rmse(true_capped, p), 2),
            "test_rmse_uncapped": round(rmse(true_rul, p), 2), "nasa_score": round(nasa_score(true_capped, p), 1),
        })
        print(rows[-1])

    final_name = "Gradient boosting, rolling + drift features"
    
    # Strict Separation: Train (80) vs Calibration (20)
    units = train["unit"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(units)
    train_units = units[:80]
    cal_units = units[80:]
    
    train_mask = train["unit"].isin(train_units)
    cal_mask = train["unit"].isin(cal_units)
    
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_cal, y_cal, groups_cal = X[cal_mask], y[cal_mask], groups[cal_mask]
    
    model = gradient_boosting().fit(X_tr, y_tr)
    lo_m = gradient_boosting("quantile", 0.1).fit(X_tr, y_tr)
    hi_m = gradient_boosting("quantile", 0.9).fit(X_tr, y_tr)
    
    pred = clip_rul(model.predict(X_test))
    raw_lo = np.minimum(clip_rul(lo_m.predict(X_test)), pred)
    raw_hi = np.maximum(clip_rul(hi_m.predict(X_test)), pred)
    raw_coverage = float(np.mean((true_capped >= raw_lo) & (true_capped <= raw_hi)))
    
    margin = calibrate_margin(lo_m, hi_m, X_cal, y_cal, groups_cal)
    
    lo = np.minimum(clip_rul(lo_m.predict(X_test) - margin), pred)
    hi = np.maximum(clip_rul(hi_m.predict(X_test) + margin), pred)
    # Notice: we no longer assert np.allclose(pred, preds[final_name]) because pred is now trained on 80 engines, not 100.
    coverage = float(np.mean((true_capped >= lo) & (true_capped <= hi)))

    sample = X_test_all.sample(3000, random_state=SEED)
    proxy = clip_rul(model.predict(sample))  # importance = how much predictions move, no labels needed
    imp = permutation_importance(model, sample, proxy, n_repeats=5, random_state=SEED, scoring="neg_root_mean_squared_error")
    by_sensor: dict[str, float] = {}
    for col, v in zip(sample.columns, imp.importances_mean):
        key = col.split("_")[0]
        by_sensor[key] = by_sensor.get(key, 0.0) + float(v)
    importance = sorted(by_sensor.items(), key=lambda kv: -kv[1])[:10]

    hist_all = clip_rul(model.predict(X_test_all))
    hist_lo = np.minimum(clip_rul(lo_m.predict(X_test_all) - margin), hist_all)
    hist_hi = np.maximum(clip_rul(hi_m.predict(X_test_all) + margin), hist_all)
    engines = []
    for i, (unit, g) in enumerate(test.groupby("unit", sort=True)):
        pos = test.index.get_indexer(g.index)
        step = max(1, len(g) // 60)
        keep = sorted(set(range(0, len(g), step)) | {len(g) - 1})
        engines.append({
            "id": int(unit), "cycles": int(g["cycle"].max()), "pred": round(float(pred[i]), 1),
            "lo": round(float(lo[i]), 1), "hi": round(float(hi[i]), 1), "true": float(true_rul[i]),
            "band": band(pred[i]), "true_band": band(true_rul[i]),
            "uncertainty": uncertainty_status(float(lo[i]), float(hi[i])),
            "risk_category": risk_category(float(pred[i]), float(lo[i]), float(hi[i])),
            "t": [int(g["cycle"].iloc[k]) for k in keep],
            "rul": [round(float(hist_all[pos[k]]), 1) for k in keep],
            "rul_lo": [round(float(hist_lo[pos[k]]), 1) for k in keep],
            "rul_hi": [round(float(hist_hi[pos[k]]), 1) for k in keep],
            "sensors": {s: [round(float(X_test_all[f"{s}_mean"].iloc[pos[k]]), 3) for k in keep] for s in TREND_SENSORS},
        })

    base = next(r for r in rows if r["model"].startswith("Constant"))
    final = next(r for r in rows if r["model"] == final_name)
    baseline_sim = simulate_fleet([e["pred"] for e in engines], set())
    
    # Phase 3 Integration: Maintenance Impact Graph
    impact_graph = build_demo_fleet_graph(engines)
    serialized_graph = [
        {
            "id": n.id, "type": n.type, "name": n.name, 
            "risk_score": n.risk_score, "risk_status": n.risk_status, 
            "parent_id": n.parent_id, "metadata": n.metadata
        }
        for n in impact_graph.nodes.values()
    ]
    
    out = {
        "dataset": {
            "name": "NASA C-MAPSS FD001", "train_engines": int(train["unit"].nunique()),
            "train_cycles": len(train), "test_engines": int(test["unit"].nunique()), "test_cycles": len(test),
            "sensors_total": 21, "sensors_used": sensors, "rul_cap": RUL_CAP, "window": WINDOW,
        },
        "models": rows, "final_model": final_name,
        "improvement_vs_constant_pct": round(100 * (1 - final["test_rmse"] / base["test_rmse"]), 1),
        "interval": {"nominal": 0.8, "coverage": round(coverage, 3), "coverage_before_calibration": round(raw_coverage, 3),
                     "conformal_margin": round(margin, 2), "mean_width": round(float(np.mean(hi - lo)), 1)},
        "alerts": [alert_quality(true_rul, pred, t) for t in (20, 30, 50)],
        "band_agreement": round(float(np.mean([e["band"] == e["true_band"] for e in engines])), 3),
        "bands": [{"name": n, "max_rul": m} for n, m in BANDS] + [{"name": "LOW", "max_rul": None}],
        "importance": [{"sensor": s, "label": SENSOR_NAMES.get(s, "Cycle count" if s == "cycle" else s), "value": round(v, 3)} for s, v in importance],
        "sensor_names": {s: SENSOR_NAMES[s] for s in TREND_SENSORS},
        "engines": engines,
        "sim_baseline": baseline_sim,
        "impact_graph": serialized_graph,
    }
    (ROOT / "docs/data.json").write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps({k: out[k] for k in ("improvement_vs_constant_pct", "interval", "alerts", "band_agreement")}, indent=1))
    print("importance", importance[:5])


if __name__ == "__main__":
    main()
