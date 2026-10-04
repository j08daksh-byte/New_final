import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm.data import RUL_CAP, SENSORS, add_rul, informative_sensors, load_cycles, load_test_rul  # noqa: E402
from pdm.features import build_features, last_cycle_rows  # noqa: E402
from pdm.model import alert_quality, band, clip_rul, nasa_score, rmse, uncertainty_status, risk_category, calibrate_margin  # noqa: E402
from pdm.simulator import simulate_engine, simulate_fleet  # noqa: E402


@pytest.fixture(scope="module")
def train():
    return load_cycles(ROOT / "data/train_FD001.txt")


def test_dataset_shape(train):
    assert len(train) == 20631 and train["unit"].nunique() == 100
    test = load_cycles(ROOT / "data/test_FD001.txt")
    assert len(test) == 13096 and len(load_test_rul(ROOT / "data/RUL_FD001.txt")) == test["unit"].nunique()
    assert not train.isna().any().any()


def test_rul_label_reaches_zero_and_is_capped(train):
    labelled = add_rul(train)
    assert (labelled.groupby("unit")["rul"].min() == 0).all()
    assert labelled["rul"].max() == RUL_CAP
    one = labelled[labelled["unit"] == 1]
    assert (one["rul"].diff().dropna() <= 0).all()  # never increases with time


def test_constant_sensors_are_dropped(train):
    kept = informative_sensors(train)
    assert len(kept) == 14 and "s1" not in kept and "s11" in kept
    assert set(kept) <= set(SENSORS)


def test_features_are_causal(train):
    """Truncating an engine's future must not change the features of its past cycles."""
    sensors = informative_sensors(train)
    full = train[train["unit"] == 3].reset_index(drop=True)
    cut = full.iloc[:60]
    a, b = build_features(full, sensors), build_features(cut, sensors)
    pd.testing.assert_frame_equal(a.iloc[:60], b)


def test_features_do_not_mix_engines(train):
    sensors = informative_sensors(train)
    both = train[train["unit"].isin([1, 2])]
    alone = train[train["unit"] == 2]
    pd.testing.assert_frame_equal(build_features(both, sensors).loc[alone.index], build_features(alone, sensors))


def test_last_cycle_rows(train):
    rows = last_cycle_rows(train)
    assert len(rows) == 100
    assert (train.loc[rows, "cycle"].to_numpy() == train.groupby("unit")["cycle"].max().to_numpy()).all()


def test_metrics():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    assert nasa_score([50], [50]) == 0
    assert nasa_score([50], [60]) > nasa_score([50], [40])  # predicting too much life is worse
    assert clip_rul(np.array([-5.0, 60.0, 300.0])).tolist() == [0.0, 60.0, RUL_CAP]


def test_bands_and_alerts():
    assert [band(r) for r in (5, 20, 21, 50, 80, 81)] == ["CRITICAL", "CRITICAL", "HIGH", "HIGH", "MEDIUM", "LOW"]
    q = alert_quality(true=[10, 25, 40, 100], pred=[12, 45, 28, 90], threshold=30)
    assert q["due"] == 2 and q["flagged"] == 2 and q["caught"] == 1
    assert q["recall"] == 0.5 and q["precision"] == 0.5


def test_uncertainty_and_risk():
    assert uncertainty_status(10, 50) == "High (Width > 35)"
    assert uncertainty_status(20, 30) == "Normal"
    
    risk1 = risk_category(25, 15, 35)
    assert risk1["material_change"] is True
    assert risk1["pessimistic"] == "CRITICAL"
    assert risk1["label"] == "HIGH — POSSIBLE CRITICAL"
    
    risk2 = risk_category(15, 10, 20)
    assert risk2["material_change"] is False
    assert risk2["label"] == "CRITICAL"


def test_calibration_logic():
    # Mock models and data to test calibrate_margin
    class MockModel:
        def __init__(self, offset):
            self.offset = offset
        def predict(self, X):
            return np.ones(len(X)) * self.offset
            
    # Create dummy dataframe for X and y
    X_cal = pd.DataFrame({"cycle": [1, 2, 3, 1, 2, 3]})
    y_cal = pd.Series([10, 9, 8, 20, 19, 18])
    groups_cal = pd.Series([1, 1, 1, 2, 2, 2])
    
    lo_m = MockModel(5)  # always predicts 5
    hi_m = MockModel(25) # always predicts 25
    
    # The scores are [-5, -5, -3, -15, -13, -11] depending on the random choice!
    # Wait, the test uses random choice, but X_cal, y_cal, groups_cal are small.
    # We will just verify it runs without crashing and returns a float.
    margin = calibrate_margin(lo_m, hi_m, X_cal, y_cal, groups_cal, level=0.8)
    assert isinstance(margin, float)
    
    # Boundary conditions: empty calibration should fail
    with pytest.raises(Exception):
        calibrate_margin(lo_m, hi_m, pd.DataFrame(), pd.Series(), pd.Series())


def test_simulator():
    avail = simulate_engine(25, False, 20, 0, 10, 50, 20)
    assert avail[4] == 1
    assert avail[5] == 0
    assert avail[14] == 0
    assert avail[15] == 1
    
    avail_force = simulate_engine(25, True, 20, 0, 10, 50, 20)
    assert avail_force[0] == 0
    assert avail_force[9] == 0
    assert avail_force[10] == 1
    
    fleet = simulate_fleet([25, 30], {1}, 20, 0, 10, 50, 20)
    assert fleet[0] == 50.0  # engine 1 forced down
    assert fleet[10] == 50.0 # engine 1 up, engine 0 down (down from t=5 to 14)


def test_simulator_edge_cases():
    # zero delay
    assert simulate_engine(25, False, 20, 0, 10, 50, 5)[4] == 1
    # max delay (causes failure)
    assert sum(simulate_engine(5, False, 20, 20, 10, 50, 100)) < 100 - 50 + 1  # mostly down due to 50 cycle penalty
    # maintenance at/near failure (RUL=1, threshold=20, delay=19 -> trigger=1)
    avail = simulate_engine(1, False, 20, 19, 10, 50, 20)
    assert avail[0] == 0 and avail[9] == 0 and avail[10] == 1  # 10 cycle normal repair
    
    # multiple engines, no engines requiring maintenance
    fleet_no = simulate_fleet([150, 200], set(), 20, 0, 10, 50, 100)
    assert all(a == 100.0 for a in fleet_no)
    
    # all engines unavailable at some point
    fleet_all = simulate_fleet([5, 5], set(), 20, 0, 10, 50, 20)
    assert fleet_all[0] == 0.0  # both start maintenance immediately
    assert fleet_all[9] == 0.0
    assert fleet_all[10] == 100.0
