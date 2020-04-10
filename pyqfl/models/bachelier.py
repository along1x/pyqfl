import math
from scipy import optimize
from scipy.stats import norm


def _getd(forward_px, strike, sigma, ttm):
    return (forward_px - strike) / (sigma * math.sqrt(ttm))


def price(forward_px, strike, sigma, ttm, rfr, is_call_option):
    d = _getd(forward_px, strike, sigma, ttm)
    put_call_scalar = 1 if is_call_option else -1
    return math.exp(-rfr * ttm) \
           * (put_call_scalar * (forward_px - strike) * norm.cdf(put_call_scalar * d) \
              + sigma * math.sqrt(ttm) * norm.pdf(d))


def delta(forward_px, strike, sigma, ttm, rfr, is_call_option):
    d = _getd(forward_px, strike, sigma, ttm)
    put_call_scalar = 1 if is_call_option else -1
    return put_call_scalar * math.exp(-rfr * ttm) * norm.cdf(put_call_scalar * d)


def gamma(forward_px, strike, sigma, ttm, rfr, is_call_option):
    d = _getd(forward_px, strike, sigma, ttm)
    return math.exp(-rfr * ttm) / (sigma * math.sqrt(ttm)) * norm.pdf(d)


def vega(forward_px, strike, sigma, ttm, rfr, is_call_option):
    d = _getd(forward_px, strike, sigma, ttm)
    return math.exp(-rfr * ttm) * math.sqrt(ttm) * norm.pdf(d)


def theta(forward_px, strike, sigma, ttm, rfr, is_call_option):
    d = _getd(forward_px, strike, sigma, ttm)
    return rfr * price(forward_px, strike, sigma, ttm, rfr, is_call_option) \
           - sigma / (2 * math.sqrt(ttm)) * math.exp(-rfr * ttm) * norm.pdf(d)


def rho(forward_px, strike, sigma, ttm, rfr, is_call_option):
    return -ttm * price(forward_px, strike, sigma, ttm, rfr, is_call_option)


def implied_vol(forward_px, strike, sigma, ttm, rfr, is_call_option, option_px):
    target_func = lambda vol: option_px - price(forward_px, strike, vol, ttm, rfr, is_call_option)
    lb, ub, xtol = 0.0001, 10.0, 0.000001
    iv = optimize.brentq(target_func, lb, ub, xtol=xtol, maxiter=100)
    return iv
