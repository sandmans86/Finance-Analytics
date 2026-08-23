## XEQT vs ZLU: Risk & Return Analysis
## https://claude.ai/code/artifact/88004711-89e9-4623-8d44-aa0fc9f6ddff


import subprocess
import sys

def ensure_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"'{package}' not found -- installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--break-system-packages', package])

ensure_packages(['pandas', 'numpy', 'yfinance', 'matplotlib', 'scipy'])

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


TICKERS = ['XEQT.TO', 'ZLU.TO']

# Your actual portfolio split -- must sum to 1.0. Edit these to match reality.
PORTFOLIO_WEIGHTS = {
    'XEQT.TO': 0.70,
    'ZLU.TO': 0.30,
}

def fetch_price_data(tickers, period='2y'):
    try:
        data = yf.download(tickers, period=period)['Close']
    except Exception as e:
        raise RuntimeError(f"Couldn't download data: {e}")
    if data.empty:
        raise RuntimeError("Couldn't download data: yfinance returned no data.")
    return data

def calculate_returns(prices):
    """
    Convert a price series into daily percentage returns.
    pct_change() computes (today - yesterday) / yesterday for each row.
    The first row has no prior day to compare to, so dropna() removes it.
    """
    return prices.pct_change().dropna()

def calculate_volatility(returns, annualize=True):
    """
    Volatility = standard deviation of returns -> how much day-to-day
    performance swings around its average, regardless of direction.
    Daily vol is annualized by multiplying by sqrt(252), the approx.
    number of trading days in a year (variance scales linearly with time,
    so std dev -- its square root -- scales with sqrt(time)).
    """
    vol = returns.std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_max_drawdown(prices):
    """
    Max drawdown = the largest peak-to-trough decline an investor would
    have experienced, as a percentage. This matters more than volatility
    for gut-check risk tolerance -- it's "how bad did it get at the worst point."

    cummax() tracks the running all-time-high price at each date.
    (price - running_high) / running_high gives the drawdown at that date
    (always <= 0). The minimum of that series is the worst drawdown.
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()


def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    """
    Sharpe ratio = risk-adjusted return: how much excess return you earned
    per unit of volatility taken on. Higher is better; above 1.0 is generally
    considered good, above 2.0 very good.

    risk_free_rate is an annual rate (default 4%, roughly a T-bill/GIC yield --
    adjust this if you want to benchmark against something else).
    Formula: (mean daily excess return / daily volatility) * sqrt(252) to annualize.
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    return (excess_returns.mean() / returns.std()) * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0.04):
    """
    Sortino ratio = risk-adjusted return, but only penalizes downside volatility.
    This is often considered a more accurate measure of risk-adjusted performance,
    since investors are typically more concerned with losses than with volatility
    in general.

    risk_free_rate is an annual rate (default 4%, roughly a T-bill/GIC yield --
    adjust this if you want to benchmark against something else).
    Formula: (mean daily excess return / daily downside deviation) * sqrt(252) to annualize.
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    downside_diff = excess_returns.clip(upper=0)
    downside_deviation = np.sqrt((downside_diff ** 2).mean())

    if downside_deviation == 0:
        return np.nan  # Avoid division by zero; Sortino ratio is undefined if no downside volatility

    return (excess_returns.mean() / downside_deviation) * np.sqrt(252)


def calculate_correlation(returns, ticker_a, ticker_b, method='pearson'):
    """
    Correlation of daily returns between two tickers, from -1 to 1.
    This is the real test of "is it actually diversified": low/negative
    correlation means the two funds tend to zig and zag independently,
    so holding both smooths the combined ride. A correlation near 1.0
    means they move together and adding the second fund barely helps --
    regardless of what their sector/geography breakdowns suggest.

    method='pearson' (default) measures the strength of a LINEAR
    relationship -- it assumes one without checking it. method='spearman'
    ranks the values instead of using them raw, so it captures any
    monotonic relationship, not just a linear one. If pearson and spearman
    come back close, the linear assumption pearson relies on is
    reasonable; a big gap between them means the relationship isn't
    actually linear and pearson is misleading.
    """
    return returns[ticker_a].corr(returns[ticker_b], method=method)


def calculate_rolling_correlation(returns, ticker_a, ticker_b, window=60):
    """
    Same pearson correlation, but recomputed on a rolling window (default
    60 trading days, ~3 months) instead of once over the whole period.
    A single full-period number can hide the fact that diversification
    tends to break down exactly when it matters most -- equity funds
    often become MORE correlated during selloffs even if they look
    diversified on average. The first `window - 1` values are NaN since
    there isn't a full window of data yet to compute over.
    """
    return returns[ticker_a].rolling(window).corr(returns[ticker_b])


def calculate_portfolio_returns(returns, weights):
    """
    Collapse per-ticker daily returns into a single blended daily return
    series, weighted by portfolio allocation (weights values should sum
    to 1.0, e.g. {'XEQT.TO': 0.7, 'ZLU.TO': 0.3} for a 70/30 split).
    Feed this into calculate_volatility/calculate_sharpe_ratio/
    calculate_max_drawdown just like a single ticker's returns.
    """
    weighted_sum = sum(returns[ticker] * weight for ticker, weight in weights.items())
    return weighted_sum


def calculate_naive_volatility(volatility_by_ticker, weights):
    """
    The volatility you'd get by weighted-averaging each ticker's vol
    directly, IGNORING how correlated they are. This is the "no
    diversification benefit" baseline -- it's what your portfolio's risk
    would be if XEQT and ZLU always moved in perfect lockstep (correlation
    = 1.0). The gap between this and the actual blended volatility
    (from calculate_volatility on calculate_portfolio_returns' output)
    is the diversification benefit, in the same annualized-vol units.
    """
    return sum(volatility_by_ticker[ticker] * weight for ticker, weight in weights.items())


def plot_correlation_check(returns, rolling_corr, ticker_a, ticker_b, window=60):
    """
    Two-panel visual check, since neither correlation number alone tells
    you the full story:

    Left: scatter of daily returns, ticker_a vs ticker_b. A cloud that
    looks roughly like a straight line supports using pearson. A cloud
    that curves, clusters, or fans out means the relationship isn't
    linear and pearson (and by extension a single correlation number)
    is understating or misdescribing it.

    Right: rolling correlation over time. Flat and stable means the
    diversification benefit has been consistent; spikes (especially
    upward, toward 1.0) show periods -- often selloffs -- where the two
    funds moved together and diversification temporarily disappeared.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(returns[ticker_a], returns[ticker_b], alpha=0.4, s=12)
    ax1.axhline(0, linewidth=0.8, color='gray')
    ax1.axvline(0, linewidth=0.8, color='gray')
    ax1.set_xlabel(f'{ticker_a} daily return')
    ax1.set_ylabel(f'{ticker_b} daily return')
    ax1.set_title('Daily return scatter (linearity check)')

    ax2.plot(rolling_corr.index, rolling_corr.values)
    ax2.axhline(0, linewidth=0.8, color='gray')
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel('Correlation')
    ax2.set_title(f'{window}-day rolling correlation')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    prices = fetch_price_data(TICKERS, period='2y')

    returns = calculate_returns(prices)
    volatility = calculate_volatility(returns)
    max_dd = calculate_max_drawdown(prices)
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    pearson_corr = calculate_correlation(returns, TICKERS[0], TICKERS[1], method='pearson')
    spearman_corr = calculate_correlation(returns, TICKERS[0], TICKERS[1], method='spearman')
    rolling_window = 60
    rolling_corr = calculate_rolling_correlation(returns, TICKERS[0], TICKERS[1], window=rolling_window)

    print("Annualized Volatility:")
    print(volatility, "\n")

    print("Max Drawdown:")
    print(max_dd, "\n")

    print("Sharpe Ratio (rf=4%):")
    print(sharpe, "\n")

    print("Sortino Ratio (rf=4%):")
    print(sortino, "\n")

    print(f"Correlation ({TICKERS[0]} vs {TICKERS[1]}):")
    print(f"  Pearson  (linear):    {pearson_corr:.4f}")
    print(f"  Spearman (monotonic): {spearman_corr:.4f}")
    print(f"  Gap: {abs(pearson_corr - spearman_corr):.4f} -- small gap supports the linear assumption pearson relies on\n")

    print(f"{rolling_window}-day rolling correlation:")
    print(f"  Min:  {rolling_corr.min():.4f} (on {rolling_corr.idxmin().date()})")
    print(f"  Max:  {rolling_corr.max():.4f} (on {rolling_corr.idxmax().date()})")
    print(f"  Mean: {rolling_corr.mean():.4f}\n")

    portfolio_returns = calculate_portfolio_returns(returns, PORTFOLIO_WEIGHTS)
    portfolio_volatility = calculate_volatility(portfolio_returns)
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    naive_volatility = calculate_naive_volatility(volatility, PORTFOLIO_WEIGHTS)
    diversification_benefit = naive_volatility - portfolio_volatility

    weights_str = ', '.join(f"{t} {w:.0%}" for t, w in PORTFOLIO_WEIGHTS.items())
    print(f"Blended portfolio ({weights_str}):")
    print(f"  Actual volatility:        {portfolio_volatility:.4f}")
    print(f"  Naive volatility (no diversification, corr=1): {naive_volatility:.4f}")
    print(f"  Diversification benefit:  {diversification_benefit:.4f} (lower risk from the two funds not moving in lockstep)")
    print(f"  Portfolio Sharpe (rf=4%): {portfolio_sharpe:.4f}")

    plot_correlation_check(returns, rolling_corr, TICKERS[0], TICKERS[1], window=rolling_window)