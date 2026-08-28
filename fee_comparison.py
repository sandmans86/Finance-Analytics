import matplotlib.pyplot as plt


STARTING_BALANCE = 10000
ANNUAL_CONTRIBUTION = 6000
GROSS_ANNUAL_RETURN = 0.07
YEARS = 30
MER_A = 0.0020   # e.g. XEQT
MER_B = 0.0180   # e.g. a typical actively-managed mutual fund


def calculate_balance_over_time(starting_balance, annual_contribution, gross_return, mer, years):
    """
    Project a portfolio balance forward year by year, net of a fee drag.
    Each year the balance grows at (gross_return - mer) instead of the
    raw gross_return -- the fee is a direct annual haircut on the growth
    rate itself, so it compounds against you every year, not just once.

    Assumption: the annual contribution is added at the END of each year,
    after that year's growth is applied -- so a given year's contribution
    doesn't start earning a return until the following year. A simplification,
    not the only valid way to model it, but a common one.
    """
    balance = starting_balance
    balances = [balance]

    net_return = gross_return - mer
    for year in range(years):
        balance = balance * (1 + net_return) + annual_contribution
        balances.append(balance)

    return balances


if __name__ == '__main__':
    balances_a = calculate_balance_over_time(STARTING_BALANCE, ANNUAL_CONTRIBUTION, GROSS_ANNUAL_RETURN, MER_A, YEARS)
    balances_b = calculate_balance_over_time(STARTING_BALANCE, ANNUAL_CONTRIBUTION, GROSS_ANNUAL_RETURN, MER_B, YEARS)

    ending_gap = balances_a[-1] - balances_b[-1]
    print(f"After {YEARS} years at {MER_A:.2%} MER: ${balances_a[-1]:,.0f}")
    print(f"After {YEARS} years at {MER_B:.2%} MER: ${balances_b[-1]:,.0f}")
    print(f"Fee drag cost you: ${ending_gap:,.0f}")

    years_axis = list(range(YEARS + 1))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years_axis, balances_a, linewidth=2, color='#2a78d6', label=f'{MER_A:.2%} MER')
    ax.plot(years_axis, balances_b, linewidth=2, color='#eb6834', label=f'{MER_B:.2%} MER')
    ax.set_xlabel('Years')
    ax.set_ylabel('Portfolio balance ($)')
    ax.set_title('Impact of fees on long-term portfolio growth')
    ax.legend()
    plt.tight_layout()
    plt.show()
