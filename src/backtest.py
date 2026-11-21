"""Long/short backtest and Sharpe ratio, shared by every strategy in notebook 02."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def long_short_backtest(prices, scores, quantile=0.05, capital=1.0, cost_bps=0.0):
    """Runs the equally weighted long/short strategy from the lecture.

    prices   : DataFrame of daily close prices (dates x stocks)
    scores   : DataFrame of predictions, one row per rebalancing date
               (a higher score means we expect the stock to do better)
    quantile : fraction of stocks to buy, and the same fraction to short (5% -> 22 of 444)
    cost_bps : optional trading cost in basis points of the money traded

    At each rebalancing date we close everything, then put half of our capital
    into the top stocks and short the same amount of the bottom stocks, equal
    money in each name. We hold the shares until the next rebalancing date.

    Returns the value of the account on every day.
    """
    dates = list(scores.index)
    values = []

    for i, date in enumerate(dates):
        start = prices.index.get_loc(date)
        # hold until the next rebalance, or to the end of the data for the last one
        stop = prices.index.get_loc(dates[i + 1]) if i + 1 < len(dates) else len(prices) - 1
        if stop <= start:
            continue

        s = scores.loc[date].dropna()
        n = int(quantile * len(s))          # int() rounds down like the lecture: 22 stocks
        longs = s.nlargest(n).index
        shorts = s.nsmallest(n).index

        # trading cost on everything we trade: close the old book and open the new one,
        # each side is worth capital / 2 (a slight overestimate, keeps it simple)
        capital -= 2 * capital * cost_bps / 1e4

        # number of shares = money / price, with capital / 2 spread equally over n names
        p0 = prices.iloc[start]
        long_shares = (capital / 2 / n) / p0[longs]
        short_shares = (capital / 2 / n) / p0[shorts]

        # account value = starting capital + gain on the longs - gain on the shorts
        # (when a shorted stock goes up we lose money, hence the minus sign)
        path = prices.iloc[start:stop + 1]
        long_value = (path[longs] * long_shares).sum(axis=1)
        short_value = (path[shorts] * short_shares).sum(axis=1)
        value = capital + (long_value - long_value.iloc[0]) - (short_value - short_value.iloc[0])

        # keep the first point only for the very first period so no day is counted twice
        values.append(value if not values else value.iloc[1:])
        capital = value.iloc[-1]

    return pd.concat(values)


def sharpe_ratio(value, every=1):
    """Annualised Sharpe ratio of an account value series.

    every=1  : uses daily returns (the usual way)
    every=20 : uses the returns between rebalancing dates, which is how the
               lecture measured it, annualised with sqrt(252 / 20)

    No risk free rate is subtracted: the strategy is dollar neutral (the
    short sale pays for the long side), so the capital just sits as cash.
    """
    v = value.iloc[::every]
    if (len(value) - 1) % every:        # make sure the final value is included
        v = pd.concat([v, value.iloc[[-1]]])
    r = v.pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS / every) * r.mean() / r.std()


def sharpe_standard_error(sharpe, n_days):
    """Rough standard error of an annualised Sharpe ratio (Lo, 2002).

    Useful sanity check: with only a few years of data the error bar is
    big, so a Sharpe of 0.8 and a Sharpe of 0.3 may not really differ.
    """
    years = n_days / TRADING_DAYS
    return np.sqrt((1 + 0.5 * sharpe**2) / years)
