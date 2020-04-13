import datetime as dt
from enum import Enum
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import pandas.tseries.holiday as holiday


class BusinessDayConvention(Enum):
    PREVIOUS = 1
    FOLLOWING = 2
    MODIFIED_PREVIOUS = 3
    MODIFIED_FOLLOWING = 4
    ACTUAL = 5


class NyseHolidayCalendar(holiday.AbstractHolidayCalendar):
    rules = [
        holiday.Holiday('NewYearsDay', month=1, day=1, observance=holiday.nearest_workday),
        holiday.USMartinLutherKingJr,
        holiday.USPresidentsDay,
        holiday.GoodFriday,
        holiday.USMemorialDay,
        holiday.Holiday('USIndependenceDay', month=7, day=4, observance=holiday.nearest_workday),
        holiday.USLaborDay,
        holiday.USThanksgivingDay,
        holiday.Holiday('Christmas', month=12, day=25, observance=holiday.nearest_workday)
    ]


class NyseTradingCalendar:
    def __init__(self):
        holidays = NyseHolidayCalendar().holidays(start=dt.datetime(1980, 1, 1),
                                                  end=dt.datetime(2099, 12, 31))
        self._holidays = set([ts.date() for ts in holidays.tolist()])

    def date_range(self, start: dt.date, end: dt.date):
        dates = []
        current_date = start
        while current_date <= end:
            dates.append(current_date)
            current_date = self.next_trading_day(current_date)
        return dates

    def is_trading_day(self, date: dt.date) -> bool:
        return date.weekday() < 5 and date not in self._holidays

    def next_trading_day(self, date: dt.date) -> dt.date:
        next_day = holiday.next_workday(date)
        while not self.is_trading_day(next_day):
            next_day = holiday.next_workday(next_day)
        return next_day

    def previous_trading_day(self, date: dt.date) -> dt.date:
        previous_day = holiday.previous_workday(date)
        while not self.is_trading_day(previous_day):
            previous_day = holiday.previous_workday(previous_day)
        return previous_day

    def modified_following_trading_day(self, date: dt.date) -> dt.date:
        next_day = self.next_trading_day(date)
        if next_day.month == date.month:
            return next_day
        return self.previous_trading_day(date)

    def modified_previous_trading_day(self, date: dt.date) -> dt.date:
        previous_day = self.previous_trading_day(date)
        if previous_day.month == date.month:
            return previous_day
        return self.next_trading_day(date)


# # adapt this to proper unittest
# cal = NyseTradingCalendar()
# d=dt.datetime(2020,4,10)
# print('Prior', cal.prior_trading_day(d))
# print('Next', cal.next_trading_day(d))
# print('N2', cal.next_trading_day(dt.date(2020, 4, 9)))
# print('P2', cal.prior_trading_day(dt.date(2020, 4, 10)))
