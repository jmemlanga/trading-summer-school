# Systematic Trading Summer School

Working through the quantitative techniques from the Imperial College Systematic
Trading Summer School (Jul 2025), implemented from scratch in Python. Each notebook
starts from the concept and builds the method up: regression and regularisation,
factor models, return forecasting with neural networks, and trading signal
construction from limit order book data.

The notebooks here are my own implementations. The course provided the problem
set-up and lecture material; the code, experiments, and write-ups are mine.

## Notebooks

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | `regression_and_regularisation` | OLS beta estimation (META vs the S&P 500), polynomial overfitting, Ridge and LASSO, time series cross validation, betas across sectors |
| 02 | `factor_models_and_return_forecasting` | Fama French 3 factor and Carhart 4 factor models on 444 S&P 500 stocks (2011 to 2023), next day return forecasting (linear, feedforward, RNN), a long/short backtest with in sample vs out of sample Sharpe ratios, sensitivity to the RNN seed, a random ranking baseline and trading costs |
| 03 | `lob_signals_logistic` | Predicting the next midprice move from limit order book data with logistic regression, degree 2 features, regularisation, order book features (spread, imbalance) and gradient boosted trees |
| 04 | `lob_signals_deep_learning` | Feedforward networks in Keras, learning rate and overfitting diagnostics, dropout, early stopping, and an out of sample comparison with the logistic models |

### Things I paid attention to

- **Same setup as the lecture, so results are comparable.** Same data files, the
  same 2019-08-31 cutoff, network sizes, seeds, and Sharpe ratio method.
- **No look ahead.** Forecasting models are split by date, never shuffled across
  time. In notebook 02 the targets are 15 day average returns, so training samples
  whose target window runs into the test period are dropped (an embargo). The
  lecture's out of sample backtest started 50 days before the cutoff (it used the
  number of training windows as a date index); notebook 02 reports both starts.
- **Scaling with training data only.** Every scaler is fitted on the training set
  and then applied to validation and test data.
- **The test set is used once.** Model choices are made on the validation set; the
  out of sample data (`Data_B`, or the last 20% of days) is only scored at the end.
- **Is the Sharpe ratio real?** Notebook 02 reports a rough standard error for
  each Sharpe ratio, compares the RNN with 200 strategies that rank stocks at
  random, and reruns the backtest with trading costs.

## Structure

```
notebooks/   the four notebooks
src/
  data.py      loading prices, Fama French factors and the order book files
  features.py  lagged windows for forecasting, order book features
  backtest.py  long/short backtest, Sharpe ratio and its standard error
data/        not committed (see data/README.md)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Python 3.10 to 3.12 recommended (TensorFlow does not support the newest Python
yet). Notebook 02 uses PyTorch and notebook 04 uses TensorFlow/Keras; both run on
a CPU (notebook 02 trains 20 RNNs, which takes a few minutes).

## Data

No data is committed. See `data/README.md` for the files each notebook expects and
where they come from.

## Next steps

- **Momentum:** backtest time series and cross sectional momentum; measure signal
  persistence via alpha decay, information ratio and Sharpe.
- **Portfolio construction:** mean variance optimal weights from a PCA denoised
  covariance matrix, with a transaction cost / turnover penalty.
