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


TICKERS = ['XEQT.TO', 'ZLU.TO', 'FEQT.NE']

# Your actual portfolio split -- must sum to 1.0. Edit these to match reality.
PORTFOLIO_WEIGHTS = {'XEQT.TO': 0.70, 'ZLU.TO': 0.30}

BENCHMARK = '^GSPC'  # S&P 500 -- lets us check ZLU's ~0.30-beta factsheet claim against real data

def fetch_price_data(tickers, period='2y'):
    try:
        data = yf.download(tickers, period=period)['Close']
    except Exception as e:
        raise RuntimeError(f"Couldn't download data: {e}")
    if data.empty:
        raise RuntimeError("Couldn't download data: yfinance returned no data.")
    return data

def calculate_correlation_matrix(returns, method = 'pearson'):
    """
    Pairwise correlation between Every pair of tickers in returns 
    at once, instad of just one named pair.  With 2 tickers this 
    is the same information as calculate correlation; with 3+ it's
    the only practical way to see the whole pircure -- an N-ticker 
    matrix has N*(N-1)/2 unique pairs, and you don't want to call 
    calculate_correlation by hand for each one.
    """
    return returns.corr(method=method)


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
    safe_downside_deviation = downside_deviation.replace(0, np.nan)  # per-ticker zero-safety; avoids ValueError from `if <Series>:`

    return (excess_returns.mean() / safe_downside_deviation) * np.sqrt(252)

def calculate_calmar_ratio(prices):
    """
    Calmar ratio = risk-adjusted return, but uses max drawdown as the risk measure.
    This is often considered a more accurate measure of risk-adjusted performance,
    since investors are typically more concerned with losses than with volatility
    in general.

    Formula: (mean annual return / max drawdown)
    """
    max_drawdown = calculate_max_drawdown(prices)
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    num_trading_days = len(prices) - 1
    cagr = (end_price / start_price) ** (252 / num_trading_days) - 1
    safe_max_drawdown = max_drawdown.replace(0, np.nan)  # per-ticker zero-safety; avoids ValueError from `if <Series>:`
    return cagr / abs(safe_max_drawdown)

def calculate_beta(returns, ticker, benchmark_returns):
    """
    Beta = how much a fund's returns move relative to a benchmark's.
    Beta of 1.0 means it moves in lockstep with the benchmark on average;
    below 1.0 (like ZLU's ~0.30 factsheet claim) means it's historically
    dampened -- moving only a fraction as much, in either direction.

    Covariance measures how the two move together; dividing by the
    benchmark's own variance rescales that into "per unit of benchmark
    movement," which is what makes beta comparable across different funds.
    Series.cov() aligns the two series by date automatically, so minor
    calendar mismatches (e.g. TSX vs. NYSE holidays) are handled for you.
    """
    covariance = returns[ticker].cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    return covariance / benchmark_variance


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


def sweep_portfolio_weights(returns, ticker_a, ticker_b, weight_b_values):
    """
    Recompute the blended portfolio's volatility, max drawdown, and Sharpe
    ratio across a range of ticker_b weights (ticker_a always gets the
    remainder, 1 - weight_b). This turns "should I hold less ZLU" from a
    one-off calculation into a curve -- so instead of an open-ended "less
    over time," you can see the specific weight where Sharpe stops
    improving as ticker_b's share shrinks, and how much drawdown
    protection you're trading away to get there.

    Returns a DataFrame, one row per weight tested, so it's easy to print
    as a table or hand straight to a plot.
    """
    rows = []
    for weight_b in weight_b_values:
        weights = {ticker_a: 1 - weight_b, ticker_b: weight_b}
        portfolio_returns = calculate_portfolio_returns(returns, weights)
        portfolio_growth = calculate_cumulative_growth(portfolio_returns)
        rows.append({
            'weight_b': weight_b,
            'volatility': calculate_volatility(portfolio_returns),
            'max_drawdown': calculate_max_drawdown(portfolio_growth),
            'sharpe': calculate_sharpe_ratio(portfolio_returns),
        })
    return pd.DataFrame(rows)


def calculate_cumulative_growth(returns):
    """
    Turn a daily return series into cumulative growth of $1 invested at
    the start of the period. (1 + returns) turns each day's pct return
    into a growth multiplier (e.g. +1% -> 1.01); cumprod() compounds each
    day's multiplier onto the running total, so the value at date T is
    what $1 would have grown to by then.
    """
    return (1 + returns).cumprod()



def plot_growth_comparison(returns, portfolio_returns, weights):
    """
    Growth of $1 over time for each standalone ticker plus the blended
    portfolio, all on the same axis -- the direct visual version of "did
    diversification's risk reduction make up for the weaker ticker
    dragging on returns." If the blended (dashed) line ends below the
    best standalone line, that fund's lower volatility didn't pay for
    itself dollar-for-dollar over this period -- the same conclusion the
    Sharpe-ratio gap already implied, just visible directly as an ending
    dollar value instead of a ratio.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2a78d6', '#eb6834', '#1baf7a']  # blue, orange, aqua -- fixed categorical order

    for (ticker, weight), color in zip(weights.items(), colors):
        growth = calculate_cumulative_growth(returns[ticker])
        ax.plot(growth.index, growth.values, linewidth=2, color=color, label=f'{ticker} (100%)')

    portfolio_growth = calculate_cumulative_growth(portfolio_returns)
    weights_str = ' / '.join(f"{t} {w:.0%}" for t, w in weights.items())
    ax.plot(portfolio_growth.index, portfolio_growth.values, linewidth=2.5,
             color=colors[2], linestyle='--', label=f'Blended ({weights_str})')

    ax.axhline(1, linewidth=0.8, color='gray')
    ax.set_ylabel('Growth of $1 invested')
    ax.set_title('Cumulative growth: standalone vs. blended portfolio')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_weight_sweep(sweep_results, ticker_b, current_weight_b):
    """
    Two-panel view of the weight sweep: Sharpe ratio and max drawdown,
    each plotted against ticker_b's portfolio weight. A dashed vertical
    line marks your current weight so you can see where you sit on both
    curves at once -- the tradeoff between "how much return am I giving
    up" (left) and "how much drawdown protection am I buying" (right)
    as ticker_b's share changes.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(sweep_results['weight_b'], sweep_results['sharpe'], linewidth=2, color='#2a78d6')
    ax1.axvline(current_weight_b, linewidth=1, color='gray', linestyle='--', label='Current weight')
    ax1.set_xlabel(f'{ticker_b} weight')
    ax1.set_ylabel('Portfolio Sharpe ratio')
    ax1.set_title('Sharpe vs. weight')
    ax1.legend()

    ax2.plot(sweep_results['weight_b'], sweep_results['max_drawdown'], linewidth=2, color='#eb6834')
    ax2.axvline(current_weight_b, linewidth=1, color='gray', linestyle='--', label='Current weight')
    ax2.set_xlabel(f'{ticker_b} weight')
    ax2.set_ylabel('Portfolio max drawdown')
    ax2.set_title('Drawdown vs. weight')
    ax2.legend()

    plt.tight_layout()
    plt.show()


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
    calmar = calculate_calmar_ratio(prices)
    benchmark_prices = fetch_price_data([BENCHMARK], period='2y').squeeze()
    benchmark_returns = calculate_returns(benchmark_prices)
    beta_by_ticker = {ticker: calculate_beta(returns, ticker, benchmark_returns) for ticker in TICKERS}
    pearson_corr = calculate_correlation(returns, TICKERS[0], TICKERS[1], method='pearson')
    correlation_matrix_pearson = calculate_correlation_matrix(returns, method='pearson')
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

    print("Calmar Ratio:")
    print(calmar, "\n")

    print(f"Beta (vs {BENCHMARK}):")
    for ticker, beta in beta_by_ticker.items():
        print(f"  {ticker}: {beta:.4f}")
    print()

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

    zlu_weight = PORTFOLIO_WEIGHTS[TICKERS[1]]
    weight_sweep_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep_results = sweep_portfolio_weights(returns, TICKERS[0], TICKERS[1], weight_sweep_values)

    print(f"\nWeight sweep ({TICKERS[1]} weight 0% to 30%, {TICKERS[0]} gets the remainder):")
    print(sweep_results.to_string(index=False))
    print("Correlation Matrix:")
    print(correlation_matrix_pearson, "\n")

    plot_correlation_check(returns, rolling_corr, TICKERS[0], TICKERS[1], window=rolling_window)
    plot_growth_comparison(returns, portfolio_returns, PORTFOLIO_WEIGHTS)
    plot_weight_sweep(sweep_results, TICKERS[1], zlu_weight)