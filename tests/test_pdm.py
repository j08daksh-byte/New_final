import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm.data import RUL_CAP, SENSORS, add_rul, informative_sensors, load_cycles, load_test_rul  # noqa: E402
from pdm.features import build_features, last_cycle_rows  # noqa: E402
from pdm.model import alert_quality, band, clip_rul, nasa_score, rmse  # noqa: E402


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
