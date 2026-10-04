# Air Power Predictive Maintenance (SIH26249 Prototype)

![tests](https://github.com/S-Harshni/Turbofan-Predictive-Maintenance/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

This prototype tackles the Ministry of Defence **SIH26249** problem statement: "Air Power - Predictive Maintenance & Fleet Availability". 
It predicts the **remaining useful life (RUL)** of jet engines from sensor readings, quantifies prediction uncertainty, and provides a fleet-level what-if simulator.

**Live demo:** `docs/index.html` (Local static dashboard)

![Maintenance queue and engine detail](docs/img/dashboard.png)

## Architecture Flow
The project is built on the NASA C-MAPSS FD001 benchmark and adds two major Unique Selling Propositions (USPs) designed for military logistics.

### 1. The ML Baseline
Predicts RUL using a Gradient Boosting architecture on 20-cycle rolling sensor features, capped at 125 cycles.

### 2. OUR USP 1: Calibrated Uncertainty-Aware Risk
Most ML systems output a single RUL number, which is dangerous in aviation. We wrap the point prediction in a conformal quantile regression interval. If the prediction interval is so wide that the lower-bound crosses a critical safety threshold, the system automatically flags a **Material Change**, overriding the optimistic point prediction and forcing human inspection.

### 3. OUR USP 2: Fleet Availability Simulator
RUL alone doesn't tell a commander if the fleet can meet a surge requirement. We decoupled the AI from policy by building a **What-If Availability Simulator**. The user can configure maintenance delays, repair times, and failure penalties to instantly project how their decisions affect long-term fleet readiness across a 100-cycle horizon.

## Validation & Results

All numbers are measured on the official held-out test set of NASA C-MAPSS FD001 (100 engines, 13,096 operating cycles). Reproduce them with `python run.py`.

| Model | CV RMSE | Test RMSE | NASA score |
| --- | ---: | ---: | ---: |
| Constant (training median) | 44.70 | 49.20 | 166,491 |
| Ridge regression, raw sensors | 19.48 | 21.32 | 1,190 |
| Gradient boosting, raw sensors | 16.50 | 17.40 | 626 |
| **Gradient boosting, rolling + drift features (FD001)** | **9.67** | **11.58** | **187** |

RMSE is in operating cycles. CV is 5-fold cross-validation grouped by engine, so no engine appears in both the training and validation folds. The NASA score penalises overestimating life more than underestimating it.

- **Feature engineering did most of the work.** The same model on engineered features has 33% lower test error than on raw sensors (17.40 → 11.58).
- **Operating-Condition Generalization:** A causal regime normalization layer was engineered to subtract baseline noise prior to temporal smoothing. This massively lowered multi-condition (FD002/FD004) RMSE by ~30% and NASA scores by ~50-60%, making the system robust across all C-MAPSS operating regimes.
- **Maintenance alerts.** Alerting when predicted RUL is 30 cycles or less catches 22 of the 25 engines that are truly within 30 cycles of failure (88% recall).
- **Priority bands.** The predicted band matches the true band for 89 of 100 engines.
- **Uncertainty.** Each prediction comes with an 80% range from quantile regression, widened by conformal calibration. The range covers the true value for ~69-81% of test engines depending on the subset, accurately reflecting the impact of real-world truncation distribution shifts.

![Accuracy, sensor importance and model comparison](docs/img/analysis.png)

## How it works

```
raw cycles ──► drop constant sensors ──► per-engine features ──► gradient boosting ──► RUL + 80% range ──► priority band ──► maintenance queue
              (21 → 14 sensors)          (rolling mean, rolling      (squared error and       (conformal          (critical ≤ 20, high ≤ 50,
                                          std, drift from the         10% / 90% quantiles)     calibration)         medium ≤ 80 cycles)
                                          engine's early life)
```

1. **Labels.** Each training engine runs to failure, so RUL is the last cycle minus the current cycle. It is capped at 125 cycles, because an engine early in life shows no degradation to learn from.
2. **Sensor selection.** Seven of the 21 sensors are constant or two-valued in FD001 and are dropped.
3. **Features.** For each remaining sensor, we apply causal regime normalization (subtracting the baseline value for its operating condition). Then we compute a 20-cycle rolling mean and standard deviation, and the *drift* of the rolling mean from the engine's own first 10 cycles. 
4. **No leakage.** Every feature uses only the current and earlier cycles of the same engine, and normalizations use strictly training-set baseline values. Tests enforce this causality.
5. **Model.** scikit-learn `HistGradientBoostingRegressor`, with two extra quantile models for the range.
6. **Decision.** Predicted RUL maps to a priority band, and the fleet is sorted by it.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                      # trains, evaluates, writes docs/data.json (a few minutes on a laptop CPU)
pytest -q                          # 8 tests
python -m http.server -d docs 8000 # dashboard at http://localhost:8000
```

## Repository layout

```
data/            NASA C-MAPSS FD001 files (unmodified) and attribution
src/pdm/         data.py (loading, labels) · features.py · model.py (models, metrics, bands)
run.py           train → evaluate → export
tests/           data integrity, label, leakage and metric tests
docs/            static dashboard (GitHub Pages) and data.json
```

## Limitations (Data Honesty)

- **Simulated Data:** C-MAPSS is **simulated** data from NASA's engine model, not readings from engines in service. Real MoD telemetry is classified and not used here.
- **Policy Assumptions:** The priority bands (critical ≤ 20), the 125-cycle cap, and the simulator repair penalties (10-cycle repair, 50-cycle failure) are project assumptions, not real MoD hangar statistics.
- **Uncertainty Calibration:** The 80% prediction interval slightly under-covers (~69%–81%) due to truncation distribution shifts between training and test sets.

## Data

A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008. NASA Ames Prognostics Center of Excellence. See [`data/README.md`](data/README.md).
