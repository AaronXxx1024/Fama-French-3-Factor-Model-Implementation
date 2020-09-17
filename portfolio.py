"""
Alpha
-----------
A Class for backtesting Fama-French Three Factor Model and present portfolio results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas_datareader import DataReader
from prettytable import PrettyTable

__author__ = 'Han (Aaron) Xiao'

class Alpha:
    """
    Alpha Strategies Execution and Performance Presentation.

    Parameters
    ----------
    :param principle: Initial cash in the account.
    :param pool: A list that contain all available tickers in the trading period.
    :param close: Close price for tickers in the pool.
    :param alpha: Alpha for all tickers in different trading day.
    :param market: Market data including S&P500 and risk free rate in the trading period.
    :param date_range: Trading period.
    :param sample: Sample length for each alpha calculation period (Regression on data from past 252 days).
    :param gap: Days between each trading day.
    :param top: How many tickers we will hold in the portfolio.
    """

    def __init__(self,
                 principle=None,
                 pool: list = None,  # ticker pool
                 close: DataFrame = None,
                 alpha: dict = None,
                 market: DataFrame = None,
                 date_range=None,
                 sample=252,
                 gap=21,
                 top: int = 0,
                 ):

        self.principle = principle
        self.pool = pool
        self.portfolio_record: dict = {}
        # An empty df that will be used to record portfolio return/value history.
        self.portfolio_value: DataFrame = DataFrame(columns=['portfolio_value', 'daily_ret', 'cumulative_return'])
        self.close = close  # if provide close price for ticker pool
        self.alpha = alpha
        self.market = market
        self.date_range = date_range  # Suggest an np.ndarray with datetime.date objects
        self.holding: dict = {}  # current holding tickers
        self.sample = sample
        self.gap = gap
        self.balance = principle  # remaining cash in the account
        self.top = top  # how many tickers we will hold in the portfolio
        self.open = None  # if provide open price for ticker pool
        self.date_list = None
        self.port_ret = None
        self.std = None
        self.sharpe_ratio = None

        if self.close is not None:
            self.close.sort_index(axis=1, inplace=True)

    # ----------------------------------------------------------------------------------------------------------

    def date_position(self):
        """
        Split date_range to separate datetime point based on regression sample and gap between each trading day.
        """
        date_position = self.date_range[self.sample::self.gap]
        date_position = list(map(str, date_position))
        self.date_list = date_position
        # return date_position

    def holding_tickers(self, top: int = None, p_matter: bool = False):
        """
        Record stocks that should be held in each trading days. The number of tickers is based on top number you choose.
        :param top: How many tickers we want to select in each alpha list.
        :param p_matter: If the p-value of alpha should be took into account. Default is False.
        :return: A list of tickers that should be included in the portfolio in the trading day.
        """
        if p_matter is False:
            output_dict = {}
            for traday in self.alpha:
                holding = list(self.alpha[traday].index.values[:top])
                output_dict.update({traday: holding})
            return output_dict
        elif p_matter is True:
            output_dict = {}
            for traday in self.alpha:
                cond = self.alpha[traday]['p_value'] <= 0.05
                holding = list(self.alpha[traday][cond].index.values[:top])
                output_dict.update({traday:holding})
            return output_dict

    def get_open(self, ticker_list: list = None, date: str = None):
        """
        Get the open price for the input ticker list at the appointed date.
        :return: A pd.df with open price for each ticker. Ticker name will be the index.
        """
        if self.open is None:
            data = DataReader(ticker_list, 'yahoo', date, date)
            output = data['Open'].T
            return output
        else:
            open_price = self.open
            return open_price

    def backtest(self, stop_date: str = None, p_matter: bool = False):
        """
        Main function that backtest F-F 3 factor alpha strategy.
        ----------------------------------------------------------------------------------------------------------------
        Trading Rules:
        Build portfolio at first key in alpha dict. Besides first trading day, if top n alpha is different with current
        holding stocks, then sell tickers needed to be replaced at open price and replace with the new one at the open
        price. Update portfolio value and return at every day, based on the close price of the holding stocks and the
        remaining cash in the account.
        ----------------------------------------------------------------------------------------------------------------
        :param stop_date: Stop backtesting at the appointed day, which must be one of the pre-designated trading day.
        :param p_matter: Choose statistically significant alpha.
        """

        if self.top <= 0:
            raise ValueError('Provide an int that means how many tickers you want to select in each trading day!')
        if p_matter is True:
            holding_tickers = self.holding_tickers(self.top, True)
        else:
            holding_tickers = self.holding_tickers(self.top)

        self.holding = {}
        initial_allocation = self.principle / self.top
        self.date_position()
        for traday in holding_tickers:
            # Stop backtesting in the appointed date
            if stop_date is not None and stop_date == traday:
                self.holding_update(traday)
                break
            # update ticker list that should be included in the portfolio in this trading day.
            ticker_update = holding_tickers[traday]
            # An empty df that will be used to record trading history.
            trading_records = DataFrame(columns=['date', 'ticker', 'activity', 'price', 'share'])
            # update portfolio return/value if current holding list is not None.
            if traday != self.date_list[0]:
                self.holding_update(traday)
            # first trading days, long all tickers in ticker_update
            if traday == self.date_list[0]:
                # update open price based on ticker list at appointed trading day.
                open_price = self.get_open(ticker_update, traday)
                for i in range(len(ticker_update)):  # ticker = ticker_update[i]
                    traday_record = [traday, ticker_update[i], 'buy', open_price.loc[ticker_update[i],traday],
                                     int(initial_allocation / open_price.loc[ticker_update[i],traday])]
                    trading_records.loc[i, trading_records.columns] = traday_record
                    self.holding.update({
                        ticker_update[i]:int(initial_allocation / open_price.loc[ticker_update[i],traday])
                    })
                cost = _sumproduct(trading_records['price'], trading_records['share'])
                self.balance = self.principle - cost
                self.portfolio_record.update(
                    {traday: {'balance': round(self.balance,2),
                              'cost':round(cost,2),
                              'holding_list': self.holding.copy(),
                              'trading_records': trading_records
                              }})
            else:
                if set(ticker_update) != set(self.holding.keys()):
                    ticker_long = [i for i in ticker_update if i not in self.holding]
                    ticker_short = [i for i in self.holding if i not in ticker_update]
                    ticker_need = ticker_short.copy()
                    ticker_need.extend(ticker_long)
                    open_price = self.get_open(ticker_need, traday)
                    self.portfolio_record.update(
                        {traday: {'balance': self.balance,
                                  'cost': 0.0,
                                  'holding_list': None,
                                  'trading_records': trading_records
                                  }})
                    # short and long
                    for short in ticker_short:
                        self.at_the_opening('sell', short, traday, open_price)
                    for long in ticker_long:
                        amount_long = len(ticker_long)
                        reallocation = self.balance / amount_long
                        self.at_the_opening('buy', long, traday, open_price, reallocation)
                    self.balance = round(self.balance,2)
                    self.portfolio_record[traday]['holding_list'] = self.holding.copy()
                    self.portfolio_record[traday]['balance'] = self.balance
                    self.portfolio_record[traday]['cost'] = round(self.portfolio_record[traday]['cost'],2)
                elif set(ticker_update) == set(self.holding):
                    pass
        self.performance()
        self.plot_cumulative_return()

    def holding_update(self, traday: str = None):
        """
        At trading day, update portfolio value and historical revenue if current holding is not None.
        """
        if len(self.holding.keys()) != 0:
            date_list = self.date_list
            pos = date_list.index(traday)
            last_traday = date_list[pos - 1]
            df = self.close[last_traday:traday].copy()
            df.drop([df.index.values[-1]], inplace=True)
            for day in df.index.values:
                holding_value = 0.0
                for ticker in self.holding:
                    holding_value += self.holding[ticker] * df.loc[day,ticker][0]
                port_value_update = holding_value + self.balance
                self.portfolio_value.loc[day, 'portfolio_value'] = port_value_update
            self.portfolio_value['daily_ret'] = self.portfolio_value['portfolio_value'].pct_change()
            self.portfolio_value.iloc[0,1] = (self.portfolio_value.iloc[0,0] / self.principle) - 1.0
            self.portfolio_value['cumulative_return'] = np.cumprod(1 + self.portfolio_value['daily_ret'])
        else:
            print('No holding stocks in the portfolio!')
            pass

    def at_the_opening(self, order: str = None, ticker:str = None, date: str = None, open_price=None,
                       reallocation=None):
        """
        In each trading day, based on input ticker name, execute 'buy' or sell activity at open price. Update account
        information accordingly.
        """
        if order == 'buy':
            share = int(reallocation / open_price.loc[ticker, date])
            self.balance = self.balance - share * open_price.loc[ticker, date]
            traday_record = [date, ticker, order, open_price.loc[ticker, date], share]
            self.portfolio_record[date]['trading_records'] = self.portfolio_record[date]['trading_records'].append(
                dict(zip(self.portfolio_record[date]['trading_records'].columns.values, traday_record)),
                ignore_index=True)
            self.holding.update({ticker:share})
            self.portfolio_record[date]['cost'] += share * open_price.loc[ticker, date]

        elif order == 'sell':
            cash = self.holding[ticker] * open_price.loc[ticker, date]
            self.balance += cash
            traday_record = [date, ticker, order, open_price.loc[ticker, date], self.holding[ticker]]
            self.portfolio_record[date]['trading_records'] = self.portfolio_record[date]['trading_records'].append(
                dict(zip(self.portfolio_record[date]['trading_records'].columns.values, traday_record)),
                ignore_index=True)
            self.holding.pop(ticker)

    def plot_cumulative_return(self):
        """
        Plot portfolio cumulative return vs. S&P500 cumulative return.
        """
        port_cum_ret = self.portfolio_value['cumulative_return']
        port_cum_ret.name = 'portfolio_cumulative_return'
        market_cum_ret = np.cumprod(1 + self.market['SP500_ret'])
        market_cum_ret.name = 'SP500_cumulative_return'
        df = pd.concat([port_cum_ret, market_cum_ret], axis=1, join='inner')
        df.plot()
        plt.legend()
        plt.grid(True)
        plt.title('Top {} alpha: Portfolio Cumulative Return vs. Market Cumulative Return'.format(self.top))
        plt.show()

    def performance(self):
        """
        :return: A table of portfolio properties.
        """
        cum_ret = (self.portfolio_value['cumulative_return'][-1] - 1) * 100
        cum_ret = round(cum_ret, 2)
        self.port_ret = cum_ret
        sharpe = round(self.sharpe(), 2)
        self.sharpe_ratio = sharpe
        volatility = round(self.volatility(),2)
        self.std = volatility
        max_drawdown = round(self.max_drawdown(),2)
        ir = self.information_ratio()
        ir = round(ir,2)

        table = PrettyTable(['Portfolio Return', 'Sharpe', 'Volatility', 'IR', 'Max Drawdown'])
        table.add_row([str(cum_ret) + '%', sharpe, volatility, ir, str(max_drawdown) + '%'])
        print(table)

    def expected_ret(self, num=252):
        """
        :return: Annualized expected return
        """
        return self.portfolio_value['daily_ret'].mean() * num

    def volatility(self, num=252):
        """
        :return: Annualized standard deviation
        """
        std = self.portfolio_value['daily_ret'].std(ddof=1)
        return std * np.sqrt(num)

    def sharpe(self, num=252):
        """
        :return: Annualized sharpe ratio.
        """
        expected_return = self.expected_ret()
        rf = self.market['daily_rf'].loc[self.portfolio_value.index.values].mean() * num
        std = self.volatility()
        sharpe = round((expected_return - rf) / std,2)
        return sharpe

    def information_ratio(self, num=252):
        expected_return = self.expected_ret()
        market_return = self.market['SP500_ret'].loc[self.portfolio_value.index.values].mean() * num
        diff = self.portfolio_value['daily_ret'] - self.market['SP500_ret'].loc[self.portfolio_value.index.values]
        relative_std = diff.std(ddof=1) * np.sqrt(num)
        ir = (expected_return - market_return) / relative_std
        return ir

    def max_drawdown(self):
        """
        :return: Max drawdown since first trading day.
        """
        value = self.portfolio_value['portfolio_value'].copy()
        drawdown = []
        for day in range(1,len(value.index)):
            drawdown.append(1-value[day]/value[:day].max())
        drawdown = pd.Series(drawdown, name='max_drawdown')
        return drawdown.max() * 100

    def total_trading_record(self):
        """
        Combine all trading records to a df.
        """
        df = DataFrame()
        for traday in self.portfolio_record:
            df = pd.concat([df, self.portfolio_record[traday]['trading_records']])
        return df

def _sumproduct(x, y):
    tmp = []
    for m, n in zip(x, y):
        tmp.append(m * n)
    return np.sum(tmp)
