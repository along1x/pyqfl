from datetime import datetime
from dateutil.relativedelta import relativedelta
from calendar import USTradingCalendar

import numpy as np

def _get_nearest_trading_day(date, businessDayConvention):
    return date


def _get_coupon_dates(valuationDate, maturityDate, couponsPerYear, businessDayConvention):
    payment_dates = [maturityDate]

    months_between_coupons = 12 / couponsPerYear
    n = 1
    while True:
        coupon_date = maturityDate - relativedelta(months=months_between_coupons * n)
        if coupon_date <= valuationDate:
            break
        coupon_date = _get_nearest_trading_day(coupon_date)
        payment_dates.append(coupon_date)
        n += 1

    payment_dates.reverse()
    return payment_dates


def _get_discount_factor(yld, t, coupon):
    return (1 + yld) ** (-t * coupon)


def price_bond(yld,
               coupon,
               coupons_per_year,
               valuation_date,
               maturity_date,
               include_accrued_interest=True,
               business_day_convention='Modified Following',
               calendar=None):
    """
    Get the price for a bond at a given valuation date
    """
    calendar = calendar if calendar is not None else USTradingCalendar()
    price = 0
    yld_per_period = yld / coupons_per_year
    coupon_per_period = coupon / coupons_per_year

    coupon_dates = _get_coupon_dates(valuation_date, maturity_date, coupons_per_year, business_day_convention)
    for coupon_date in coupon_dates:
        t = (coupon_date - valuation_date).days / 365.0
        discount_factor = _get_discount_factor(yld_per_period, t, coupons_per_year)
        price += coupon_per_period * discount_factor

    # add the discounted face value received at maturity
    price += 1 * _get_discount_factor(yld_per_period, (maturity_date - valuation_date).days / 365.0, coupons_per_year)

    coupon_dates = [datetime()]
    if not include_accrued_interest:
        time_to_next_coupon = (coupon_dates[0] - valuation_date).days / 365.0
        time_since_last_coupon = 1 / coupons_per_year - time_to_next_coupon
        accrued_interest = coupon * time_since_last_coupon
        price -= accrued_interest

    return price

# def modifiedDuration():


def pv(cashflows, r):
    """
    Compute the present value of a sequence of cash flows
    """
    dates = cashflows.index
    discount_factors = discount_factor(dates, r)
    return (discount_factors * cashflows).sum()


def get_annual_rate(r_inst):
    """
    Converts the short rate to an annualized rate
    """
    return np.expm1(r_inst)


def get_short_rate(r_ann):
    """
    Converts annualized rate to a short rate
    """
    return np.log1p(r_ann)
