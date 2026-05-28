# Indian CPI Inflation Forecasting — P2

Macro-quantitative forecasting pipeline for Indian CPI inflation
using walk-forward validation across four model families.

## Research Question

Can machine learning models extract predictive information from
macro variables that pure autoregression misses? And if so, does
combining both approaches produce better forecasts than either alone?

## Dataset

172 monthly observations, January 2012 to April 2026.
Source: RBI DBIE, FRED, MoSPI.

Variables: CPI inflation, repo rate, M3, IIP, 91d T-bill,
10yr G-sec, INR/USD, term spread, Fed funds rate, Brent crude oil.

## Models and Results

| Model | RMSE | MAE |
|---|---|---|
| ARIMA(3,0,0) | 0.7024 | 0.5397 |
| ARIMA-XGBoost Hybrid | 0.7044 | 0.5316 |
| Elastic Net | 0.7213 | 0.5458 |
| ARIMAX(3,0,0) | 0.7246 | 0.5438 |
| XGBoost | 0.8390 | 0.6607 |

Test window: January 2022 to April 2026.
All forecasts produced using expanding window walk-forward
validation, no future information used at any step.

## Key Findings

**ARIMA dominates on RMSE.** AR(3) captures Indian inflation
remarkably well as a pure autoregressive process. Three months
of inflation persistence explains most of the forecastable
variation, a direct consequence of sticky prices, expectation
formation, and supply chain propagation.

**XGBoost standalone underperforms.** Despite SHAP-validated
feature selection and extensive hyperparameter tuning, XGBoost
could not beat AR(3). 108 training observations is a hard
constraint for tree-based models. External macro relationships
are time-varying, sometimes crude oil dominates, sometimes M3,
sometimes neither, and XGBoost struggles to learn switching
behavior on limited data.

**Hybrid is the best model on MAE.** ARIMA captures linear
autoregressive structure. XGBoost corrects residual errors
using crude oil, exchange rate, and M3. Together they produce
more consistent forecasts than either alone — MAE 0.5316
versus ARIMA 0.5397.

**SHAP independently confirmed P1 monetary transmission finding.**
M3 lag 12 ranks fourth in global feature importance. SHAP
identified the same 12-month monetary transmission channel
that the VECM in P1 found through cointegration analysis,
two completely different methodologies converging on the same
economic mechanism.

**Crude oil spike at Russia-Ukraine 2022.**
SHAP time-varying contributions show crude oil lag 1 spiking
precisely at February 2022, the model identified the supply
shock timing without being told. This cross-validates the
rolling correlation finding from EDA independently.

## Project Structure

scripts/
data_loader.py      — loads data, stationarity tests, lag features
validator.py        — walk-forward validation for sklearn and statsmodels
evaluator.py        — stores results, comparison table, plots
notebooks/
01_eda.ipynb        — exploratory analysis, rolling correlations
02_stationarity.ipynb — ADF KPSS tests, feature engineering
03_arima.ipynb      — ARIMA and ARIMAX baseline
04_xgboost.ipynb    — XGBoost with hyperparameter tuning, hybrid model
05_elastic_net.ipynb — Elastic Net walk-forward
06_shap.ipynb       — SHAP global importance, beeswarm, time-varying
07_model_comparison.ipynb — full pipeline, all models, final results

## How To Run

```bash
pip install -r requirements.txt
jupyter lab --notebook-dir=F:/
```

Run notebooks in order 01 through 07.

## Related Work

P1 — Monetary Policy Transmission in India: A VAR vs VECM Analysis
SHAP findings in this project independently confirm the 12-month
M3 transmission channel identified in P1 through cointegration.