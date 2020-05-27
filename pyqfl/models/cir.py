import numpy as np
import pandas as pd

from bond_math import get_short_rate, get_annual_rate

def cir(
    n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None
):
    """
    Implement Cox-Ingersoll-Rand interest rate model
    r_0 and b are assumed to be annualized rates
    """
    if r_0 is None:
        r_0 = b

    r_0 = get_short_rate(r_0)  # convert annualized to instantaneous rate
    dt = 1 / steps_per_year

    num_steps = int(n_years * steps_per_year) + 1  # first element is the current rate
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(
        shock
    )  # gets me a new array with the same dimensions as my shocks, but everything is None
    rates[0] = r_0

    ## for price generation
    h = np.sqrt(a ** 2 + 2 * sigma ** 2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        A = (
            (2 * h * np.exp((a + h) * ttm / 2))
            / (2 * h + (a + h) * (np.exp(ttm * h) - 1))
        ) ** (2 * a * b / sigma ** 2)
        B = 2 * (np.exp(ttm * h) - 1) / (2 * h + (a + h) * (np.exp(ttm * h) - 1))
        return A * np.exp(-B * r)

    prices[0] = price(n_years, r_0)
    ##

    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        # ensure that the rates are always positive - shouldn't be negative, but possible with floating point precision issues
        rates[step] = abs(r_t + d_r_t)

        ## generate prices for all times t as well
        prices[step] = price(n_years - step * dt, rates[step])

    rates = pd.DataFrame(data=get_annual_rate(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))

    return rates, prices
