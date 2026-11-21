# Data

Data is **not** committed to this repo (it is course material). The course
provides it as one zip, `data_full.zip`; the download link is in the course
notebooks. Unzip it into this folder: either `data/` or `data/data_full/` works,
the loaders in `src/data.py` look in both.

| File | Used in | What it is |
|------|---------|------------|
| `sp500stocks.csv` | 02 | Daily adjusted close prices of 444 S&P 500 stocks, 2011 to 2023 (dates in the first column, one column per stock) |
| `F-F_Research_Data_Factors_daily.csv` | 02 | Ken French data library, Fama/French 3 factors, daily (raw CSV as downloaded) |
| `F-F_Momentum_Factor_daily.csv` | 02 | Ken French data library, momentum factor, daily |
| `Data_A.csv` | 03, 04 | 100,000 limit order book snapshots, 22 columns, no header (train + validation) |
| `Data_B.csv` | 03, 04 | 10,000 more snapshots drawn the same way (out of sample test) |

Notebook 01 downloads its own data (META from `yfinance`, the S&P 500 from FRED).

If `sp500stocks.csv` is missing, notebook 02 falls back to downloading today's
S&P 500 constituents with `yfinance` (slower, and the results will differ from
the course). If the Ken French files are missing they are downloaded with
`pandas_datareader`.
