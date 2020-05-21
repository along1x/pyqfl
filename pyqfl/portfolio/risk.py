import numpy as np
import pandas as pd
import scipy.stats as stats


def as_index(return_series: pd.Series, scale=1000) -> pd.Series:
    """Take a series of returns and translate it to a set of index levels"""
    return scale * (1 + return_series).cumprod()


def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """
    Takes a time series of asset returns and computes an index, previous
    peaks, and % drawdown
    """
    index_levels = as_index(return_series)
    previous_peaks = index_levels.cummax()
    drawdowns = (index_levels - previous_peaks) / previous_peaks
    return drawdowns


def max_drawdown(return_series: pd.Series) -> float:
    return drawdown(return_series).max()


def is_normal(return_series, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series ir normal or not
    Test is applied at 1% level by default
    Returns True if the hypothesis (is normal) is accepted, False otherwise
    """ 
    _, p_value = stats.jarque_bera(return_series)
    return p_value > level


def historical_var(pd_obj, percentile=5):
    """
    Returns the historical VaR at a specified percentile

    Args:
        pd_obj (DataFrame or Series): The pandas object
        percentile (float): Percentile (e.g. 5% would be 5, not 0.05)
    """
    if isinstance(pd_obj, pd.DataFrame):
        return pd_obj.aggregate(historical_var, percentile=percentile)
    elif isinstance(pd_obj, pd.Series):
        return -np.percentile(pd_obj, percentile)
    else:
        raise TypeError('pd_obj must be Series or DataFrame')


def gaussian_var(pd_obj, percentile=5, modified=False):
    """
    Return the Parametric Gaussian VaR of a Series or DataFrame

    If modified == True, then modified VaR is returned
    using the Cornish-Fisher modification
    """
    # compute the z score, assuming a Gaussian distribution
    z = stats.norm.ppf(percentile / 100)

    if modified:
        skew = stats.skew(pd_obj)
        excess_kurtosis = stats.kurtosis(pd_obj)
        z += (z**2 - 1) * skew / 6 \
             + (z**3 - 3*z) * excess_kurtosis / 24 \
             - (2*z**3 - 5*z) * z**2 / 36

    return -(pd_obj.mean() + z*pd_obj.std(ddof=0))


def historical_cvar(pd_obj, percentile=5):
    """
    Compute the Conditional Value at Risk of a Series or DataFrame
    using the Historical VaR methodology
    """
    if isinstance(pd_obj, pd.Series):
        return -pd_obj[pd_obj <= -historical_var(pd_obj, percentile=percentile)].mean()
    elif isinstance(pd_obj, pd.DataFrame):
        return pd_obj.aggregate(historical_var, percentile=percentile)
    else:
        raise TypeError('r must be Series or DataFrame')
