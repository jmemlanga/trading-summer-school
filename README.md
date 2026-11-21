# Systematic Trading Summer School

My implementations of the techniques from Imperial College's Systematic Trading
Summer School (Jul 2025). The course provided the problem set-up and data.

| # | Notebook | Covers |
|---|----------|--------|
| 01 | `regression_and_regularisation` | OLS beta, overfitting, Ridge and LASSO, time series cross validation |
| 02 | `factor_models_and_return_forecasting` | Fama French and Carhart factor models, RNN return forecasting, long/short backtest and Sharpe ratios |
| 03 | `lob_signals_logistic` | Limit order book direction prediction with logistic regression and gradient boosted trees |
| 04 | `lob_signals_deep_learning` | The same task with Keras networks: learning rate, dropout, early stopping |

Notebook 02 also tests whether the Sharpe ratio is real: no look ahead (an embargo
at the train/test cutoff), standard errors, a random ranking baseline and trading costs.

Shared code lives in `src/` (data loading, features, backtest).

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 to 3.12. Data is not committed; see `data/README.md`.
