"""Turning daily data into the (window -> target) samples the models learn from."""

import numpy as np


def make_windows(features, targets, window, horizon=1):
    """Builds lagged samples for forecasting.

    features : array (T, F), one row per day
    targets  : array (T, S), daily returns of each stock
    window   : how many past days each sample looks at
    horizon  : the target is the MEAN return over the next `horizon` days

    Sample i uses days t-window+1 ... t as input (t is the last day we know)
    and the mean of days t+1 ... t+horizon as the target, so nothing from the
    future leaks into the input.

    Returns X (N, window, F), y (N, S) and t (N,), the position of the last
    input day for each sample (we need it to split by date and to trade).
    """
    T = len(features)
    X, y, t_idx = [], [], []
    for t in range(window - 1, T - horizon):
        X.append(features[t - window + 1:t + 1])
        y.append(targets[t + 1:t + 1 + horizon].mean(axis=0))
        t_idx.append(t)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(t_idx)


def window_ending_at(features, t, window):
    """The single input window whose last day is t (used on trading days)."""
    return features[t - window + 1:t + 1]


def add_lob_features(df):
    """Adds a few market microstructure features to the limit order book data.

    The raw prices are absolute levels (they drift over the three months), so
    on their own they say little about the NEXT move. What traders actually
    look at is the shape of the book:

    spread        : ask - bid, how wide the market is
    imbalance_1   : (bid vol - ask vol) / (bid vol + ask vol) at level 1,
                    +1 means all the size is on the bid (buyers queueing)
    imbalance_all : the same using the total volume over all 4 levels
    ask_gap, bid_gap : distance from level 1 to level 2 on each side. A big
                    gap means the book is thin behind the best price.

    I also tried the volume weighted "microprice" minus the mid, but it turns
    out to be exactly (spread / 2) x imbalance_1, so it adds nothing new.
    """
    out = df.copy()
    eps = 1e-9  # avoids dividing by zero if a level is ever empty

    out["spread"] = df["ask1_price"] - df["bid1_price"]
    out["imbalance_1"] = (df["bid1_vol"] - df["ask1_vol"]) / (df["bid1_vol"] + df["ask1_vol"] + eps)

    bid_depth = sum(df[f"bid{k}_vol"] for k in range(1, 5))
    ask_depth = sum(df[f"ask{k}_vol"] for k in range(1, 5))
    out["imbalance_all"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + eps)

    out["ask_gap"] = df["ask2_price"] - df["ask1_price"]
    out["bid_gap"] = df["bid1_price"] - df["bid2_price"]
    return out
