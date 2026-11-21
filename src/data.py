"""Loading the data used in the notebooks.

The course data comes as one zip (data_full.zip) with these files:
Data_A.csv, Data_B.csv, sp500stocks.csv, F-F_Research_Data_Factors_daily.csv,
F-F_Momentum_Factor_daily.csv (and a few extras we don't use).

Unzip it into the data/ folder. Either data/ or data/data_full/ works, the
loaders look in both.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# data/ sits one level up from src/, at the root of the repo
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEARCH_DIRS = [DATA_DIR, DATA_DIR / "data_full"]


def find_file(*names):
    """First file that exists among the given names, in data/ or data/data_full/."""
    for folder in SEARCH_DIRS:
        for name in names:
            if (folder / name).exists():
                return folder / name
    return None


# ---------------------------------------------------------------------------
# Daily stock prices (notebook 02)
# ---------------------------------------------------------------------------

def load_prices(start="2011-01-01", end="2023-12-31"):
    """Daily adjusted close prices, one column per stock, dates as the index.

    Uses the course file sp500stocks.csv (444 S&P 500 stocks, 2011 to 2023).
    If that is missing it falls back to data/prices.csv, and if that is missing
    too it downloads the current S&P 500 with yfinance and saves prices.csv.
    """
    path = find_file("sp500stocks.csv", "prices.csv")

    if path is not None:
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        prices = _download_sp500_prices(start, end)
        DATA_DIR.mkdir(exist_ok=True)
        prices.to_csv(DATA_DIR / "prices.csv")

    prices = prices.sort_index().loc[start:end]

    # a stock only survives if it has a price on every single day
    # (otherwise the windows we build later would have holes in them)
    prices = prices.dropna(axis=1, how="any")
    return prices


def _download_sp500_prices(start, end):
    # only imported here so the rest of the repo works without yfinance
    import yfinance as yf

    # current constituents from Wikipedia
    # NOTE: this is today's list, so it has survivorship bias (firms that
    # dropped out of the index over the 13 years are missing)
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw["Close"]


# ---------------------------------------------------------------------------
# Fama French and momentum factors (notebook 02)
# ---------------------------------------------------------------------------

FF3_FILE = "F-F_Research_Data_Factors_daily.csv"
MOM_FILE = "F-F_Momentum_Factor_daily.csv"


def load_factors():
    """Daily Mkt-RF, SMB, HML, RF and Mom, in PERCENT simple returns.

    Reads the two Ken French daily files from data/ if they are there,
    otherwise downloads them with pandas_datareader.
    """
    ff3 = _load_french(FF3_FILE, "F-F_Research_Data_Factors_daily")
    mom = _load_french(MOM_FILE, "F-F_Momentum_Factor_daily")

    # the momentum file calls its column "Mom   " with trailing spaces
    mom.columns = ["Mom"]
    factors = ff3.join(mom, how="inner")
    return factors


def _load_french(filename, dataset_name):
    path = find_file(filename, filename.replace(".csv", ".CSV"))
    if path is not None:
        return _read_french_csv(path)

    import pandas_datareader.data as web
    df = web.DataReader(dataset_name, "famafrench", start="1990-01-01")[0]
    df.index = pd.to_datetime(df.index)
    df.columns = [c.strip() for c in df.columns]
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path)
    return df


def _read_french_csv(path):
    """Reads either the raw Ken French CSV or a plain csv we saved ourselves.

    The raw file has a few lines of text at the top and a copyright line at
    the bottom, so we only keep rows whose first field is an 8 digit date.
    """
    rows, header = [], None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if header is None and len(parts) > 1 and parts[0] in ("", "Date"):
                header = parts[1:]
                continue
            if header is not None and len(parts) == len(header) + 1:
                first = parts[0].replace("-", "")[:8]
                if first.isdigit() and len(first) == 8:
                    rows.append([first] + parts[1:])

    df = pd.DataFrame(rows, columns=["Date"] + header)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").astype(float)
    return df


# ---------------------------------------------------------------------------
# Limit order book data (notebooks 03 and 04)
# ---------------------------------------------------------------------------

LOB_COLUMNS = (
    ["label"]
    + [f"{side}{lvl}_{kind}" for lvl in range(1, 5)
       for side, kind in [("ask", "price"), ("ask", "vol"), ("bid", "price"), ("bid", "vol")]]
    + [f"prev_dir_{k}" for k in range(1, 6)]
)


def load_lob(name):
    """Loads Data_A or Data_B as a DataFrame with readable column names.

    Column order follows the course description: label, then four levels of
    (ask price, ask volume, bid price, bid volume), then the five previous
    midprice directions.
    """
    path = find_file(f"{name}.csv")
    if path is None:
        raise FileNotFoundError(
            f"{name}.csv not found. Unzip the course data_full.zip into the data/ folder."
        )

    df = pd.read_csv(path, header=None)
    # if the first row is text then the file has a header row we should skip
    if not np.issubdtype(df.dtypes.iloc[0], np.number):
        df = pd.read_csv(path, header=0)

    df.columns = LOB_COLUMNS[: df.shape[1]]
    return df
