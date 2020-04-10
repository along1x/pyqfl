import math
from scipy.stats import norm
from scipy import optimize


def _getd1d2(spot, strike, ttm, sigma, rfr, div):
    d1 = (math.log(spot / strike) + (rfr - div + 0.5 * sigma * sigma) * ttm) \
         / (sigma * math.sqrt(ttm))
    d2 = d1 - sigma * math.sqrt(ttm)
    return d1, d2


def price(spot, strike, ttm, sigma, rfr, div, isCallOption):
    d1, d2 = _getd1d2(spot, strike, ttm, sigma, rfr, div)

    call_px = spot * math.exp(-div * ttm) * norm.cdf(d1) \
              - strike * math.exp(-rfr * ttm) * norm.cdf(d2)
    if isCallOption:
        return max(call_px, 0)
    else:
        put_px = call_px - spot * math.exp(-div * ttm) + strike * math.exp(-rfr * ttm)
        return max(put_px, 0)


def delta(spot, strike, ttm, sigma, rfr, div, is_call_option):
    d1, d2 = _getd1d2(spot, strike, ttm, sigma, rfr, div)

    call_delta = math.exp(-div * ttm) * norm.cdf(d1)
    if is_call_option:
        return call_delta
    else:
        return call_delta - math.exp(-div * ttm)


def gamma(spot, strike, ttm, sigma, rfr, div):
    d1, _ = _getd1d2(spot, strike, ttm, sigma, rfr, div)
    return math.exp(-div * ttm) * norm.pdf(d1) / (spot * sigma * math.sqrt(ttm))


def speed(spot, strike, ttm, sigma, rfr, div):
    d1, _ = _getd1d2(spot, strike, ttm, sigma, rfr, div)
    phid1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return -math.exp(-div * ttm) * phid1 \
           / (spot * spot * sigma * math.sqrt(ttm)) \
           * (d1 / (sigma * math.sqrt(ttm)) + 1)


def vega(spot, strike, ttm, sigma, rfr, div):
    d1, _ = _getd1d2(spot, strike, ttm, sigma, rfr, div)
    return spot * math.sqrt(ttm) * math.exp(-div * ttm) * norm.pdf(d1)


def theta(spot, strike, ttm, sigma, rfr, div, is_call_option):
    d1, d2 = _getd1d2(spot, strike, ttm, sigma, rfr, div)

    call_theta = div * spot * math.exp(-div * ttm) * norm.cdf(d1) \
                 - rfr * strike * math.exp(-rfr * ttm) * norm.cdf(d2) \
                 - spot * sigma * math.exp(-div * ttm) / (2 * math.sqrt(ttm)) * norm.pdf(d1)

    if is_call_option:
        return call_theta
    else:
        return call_theta - div * spot * math.exp(-div * ttm) + rfr * strike * math.exp(-rfr * ttm)


def rho(spot, strike, ttm, sigma, rfr, div, is_call_option):
    d1, d2 = _getd1d2(spot, strike, ttm, sigma, rfr, div)

    call_rho = ttm * strike * math.exp(-rfr * ttm) * norm.cdf(d2)

    if is_call_option:
        return call_rho
    else:
        return call_rho - ttm * strike * math.exp(-rfr * ttm)


def dpx_ddiv(spot, strike, ttm, sigma, rfr, div, is_call_option):
    d1, d2 = _getd1d2(spot, strike, ttm, sigma, rfr, div)

    call_dpx_ddiv = -math.sqrt(ttm) / sigma * spot * math.exp(-div * ttm) * norm.pdf(d1) \
                    - spot * ttm * math.exp(-div * ttm) * norm.cdf(d1) + strike * math.sqrt(ttm) / sigma \
                    * math.exp(-rfr * ttm) * norm.pdf(d2)

    if is_call_option:
        return call_dpx_ddiv
    else:
        return call_dpx_ddiv + spot * ttm * math.exp(-div * ttm)


def implied_vol(spot, strike, ttm, rfr, div, is_call_option, option_px):
    target_func = lambda vol: option_px - price(spot, strike, ttm, vol, rfr, div, is_call_option)
    lb, ub, xtol = 0.00001, 10.0, 0.00001
    return optimize.brentq(target_func, lb, ub, xtol=xtol, maxiter=100)


def implied_div(spot, strike, ttm, sigma, rfr, is_call_option, option_px):
    target_func = lambda div: - price(spot, strike, ttm, sigma, rfr, div, is_call_option)
    lb, ub, xtol = -0.3, 0.3, 0.00001
    return optimize.brentq(target_func, lb, ub, xtol=xtol, maxiter=100)
