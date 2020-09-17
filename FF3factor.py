"""
Fama-French-Three-Factor-Model Implementation
----------------------------------------------------------------------------------------------------------------
Here is the workflow of the model realization in this file:
1) Get data (ticker pool, S&P500, risk free rate, close price, market cap and book-to-market ratio) from SQL database.
2) Organize data to the form we want: {trading_day:df_data}
3) Calculate factor in two different ways:
    i. Split mc to 'Big'(50%) and 'Small'(50%), then use market cap weighted average return ('Small'-'Big') to get 'SMB'
    Split bm to 'High'(30%), 'Medium'(40%) and 'L'(30%), then use market cap weighted average return ('High'-'Low') to
    get 'HML'.
    ii. The difference is after initial separation, we do one more step. Mark tickers based on intersection, e.g. if a
    ticker is marked as 'Big' and 'High' in the same time, then we mark it as 'B/H'. Therefore, we'll have total 6
    different groups: B/H, B/M, B/L, S/H, S/M and S/L. Finally, use market cap weighted average return
    ((S/H + S/M + S/L) / 3 - (B/H + B/M + B/L) / 3) to get 'SMB' and use market cap weighted average return
    ((B/H + S/H) / 2 - (B/L + S/l) / 2) go get 'HML'.
4) Save all factor data in a df with columns ['Rm', 'SMB', 'HML'].
   where 'Rm' is the log return of S&P500 minus corresponding daily risk free rate.
5) Regress all tickers' log return on factor data, get interception as 'alpha' and its p-value. Save these two data to
   a dict called 'alpha' with form: {trading_day:df_data([alpha, p-value])}.
6) Input all necessary data to a Class to get an 'Alpha' object.
7) Run backtest() method in Alpha object. When backtesting is done, program will plot portfolio cumulative return vs.
   market cumulative return and print the portfolio result in a table form with columns:
   [Portfolio Return | Sharpe | Volatility |  IR  | Max Drawdown].

   e.g Top 25 alpha: Portfolio results
   +------------------+--------+------------+------+--------------+
   | Portfolio Return | Sharpe | Volatility |  IR  | Max Drawdown |
   +------------------+--------+------------+------+--------------+
   |      38.95%      |  0.68  |    0.17    | 0.19 |    21.4%     |
   +------------------+--------+------------+------+--------------+

   The detailed info about trading rules and Alpha Class could be found in portfolio.py
----------------------------------------------------------------------------------------------------------------
"""

__author__ = 'Han (Aaron) Xiao'

import pandas as pd
import numpy as np
import statsmodels.api as sm
import portfolio as pf
import pymysql

# ---------------------------------------------------------------------
# Set necessary environment
db = pymysql.connect(host='localhost', user='root', password="",
                     database='ff_3factor', port=3308,
                     charset='utf8')

# ---------------------------------------------------------------------
# Functions: Prepare data for factor calculation
def get_ticker(connection=None):
    """
    Get a list of all tickers we have from sql sever.
    """
    sql = 'SELECT `Ticker` FROM `ticker_list`'
    df = pd.read_sql(sql, connection)
    df.sort_values(by=['Ticker'], inplace=True)
    name_list = list(df['Ticker'])
    return name_list

def get_market(connection=None):
    """
    Get time benchmark, market data (S&P500) and risk free rate.
    """
    sql = 'SELECT `date`, `S&P500`, `risk_free` FROM `marketdata`'
    df = pd.read_sql(sql, connection)
    df.sort_values(by=['date'])
    return df

def get_close(tickerlist, daterange, connection=None):
    """
    Get daily close price based on input ticker list from sql server. Save them into a dict.
    """
    output_dict = {}
    for ticker in tickerlist:
        sql = "SELECT `date`, `close` FROM `stockdata` WHERE ticker = '{}'".format(ticker)
        df = pd.read_sql(sql, connection)
        df['date'] = df['date'].astype('datetime64')
        df.set_index(keys=['date'], inplace=True)
        df.sort_index(axis=0, inplace=True)
        df.fillna(method='ffill', inplace=True)
        output_dict.update({ticker:df})
    output_df = pd.concat(output_dict, axis=1)
    output_df = __position_check(output_df, daterange)
    return output_df

def get_ret(input_df, daterange=None, column=None, log=False):
    """
    Calculate daily ret/log_ret for each stock/index. Save them in input dict and update the input df.
    """
    if column == 'close':
        output_df = input_df.pct_change()
        if log is True:
            output_df = np.log(1 + output_df)
        output_df.dropna(inplace=True)
        output_df = __position_check(output_df, daterange)
        return output_df

    elif column == 'S&P500':
        if isinstance(input_df,pd.DataFrame):
            input_df['SP500_ret'] = input_df[column].pct_change()
            input_df['SP500_log_ret'] = np.log(1 + input_df['SP500_ret'])
            input_df.dropna(inplace=True)
            input_df['daily_rf'] = input_df['risk_free'] / 100 / 252
            input_df['Rm'] = input_df['SP500_ret'] - input_df['daily_rf']
            input_df['log_Rm'] = input_df['SP500_log_ret'] - input_df['daily_rf']
        input_df['date'] = input_df['date'].astype('datetime64')
        input_df.set_index(keys=['date'], inplace=True)
        input_df.sort_index(axis=0, inplace=True)
        output_dict = position_split(input_df, daterange)
        return output_dict

def get_marketcap(tickerlist, daterange, connection=None,):
    """
    Get daily market capital based on input ticker list from sql server. Save them into a dict.
    """
    output_dict = {}
    for ticker in tickerlist:
        sql = "SELECT `date`, `market_value` FROM `stockdata` WHERE ticker = '{}'".format(ticker)
        df = pd.read_sql(sql, connection)
        df['date'] = df['date'].astype('datetime64')
        df.set_index(keys=['date'], inplace=True)
        df.sort_index(axis=0, inplace=True)
        output_dict.update({ticker: df})
    output_df = pd.concat(output_dict, axis=1)
    output_df = __position_check(output_df, daterange)
    return output_df

def get_bm(tickerlist, daterange, connection=None):
    """
    Get daily book-to-market ration based on input ticker list from sql server. Save them into a dict.
    """
    output_dict = {}
    for ticker in tickerlist:
        sql = "SELECT `date`, `book_to_market` FROM `stockdata` WHERE ticker = '{}'".format(ticker)
        df = pd.read_sql(sql, connection)
        df['date'] = df['date'].astype('datetime64')
        df.set_index(keys=['date'], inplace=True)
        df.sort_index(axis=0, inplace=True)
        output_dict.update({ticker: df})
    output_df = pd.concat(output_dict, axis=1)
    output_df = __position_check(output_df, daterange)
    return output_df

def position_split(data_df, date_range, sample_length=252, trading_gap=21):
    """
    Split a df (could be market data or stock data) to different keys in a dict, based on trading days.
    :param data_df: df with 1006 index about days and with 498 columns about tickers.
    :param date_range: a ndarray that consist of total 1006 datetime.date objects, will be used to create trading days.
    :param sample_length: sample length for regression to get alpha
    :param trading_gap: gap between each trading days
    :return: return a dict consist of different trading days (key) and their corresponding past 252 days data records.
    """
    output_dict = {}
    date_position = date_range[sample_length::trading_gap]

    for i in range(len(date_position)):
        date_title = str(date_position[i])
        right = __date_position(date_range, date_title)
        date_left = date_range[right-sample_length]
        date_right = date_range[right-1]
        df = data_df[date_left:date_right]
        output_dict.update({date_title:df})

    return output_dict

def get_factor_mark(position_dict, catog=None):
    """
    Return an exact same size dict like input position_dict, with exact same size df in each keys.
    Based on tickers' performance in each day, mark them accordingly.
    ----------------------------------------------------------------------------------------------------------------
    Rules:
    1) In mc_dict, mark 'B' to tickers whose market capital is in the top 50%, mark 'S' to the rest of tickers.
    2) In bm_dict, mark 'H' to tickers whose book-to-market ratio is in the top 30%, mark 'L' to the bottom 30%, mark
    'M' to the rest of them (40%).
    ----------------------------------------------------------------------------------------------------------------
    """
    output_dict = {}
    if catog == 'mc':
        for trading_day in position_dict:
            mc_mark = pd.DataFrame().reindex_like(position_dict[trading_day])
            position_df = position_dict[trading_day].copy().T
            # Total number of days used for collecting data before each trading days
            amount_days = len(position_df.columns)
            # Total number of tickers
            amount_tickers = len(position_df.index)
            for day in range(amount_days):
                mc_daily = position_df.iloc[:,day].copy()
                mc_daily.sort_values(inplace=True)
                big = mc_daily[int(amount_tickers/2):].index.values
                small = mc_daily[:int(amount_tickers/2)].index.values
                mark_day = mc_mark.index[day]
                for mark in big:
                    if mark in mc_mark.columns:
                        mc_mark.loc[mark_day, mark] = 'B'
                for mark in small:
                    if mark in mc_mark.columns:
                        mc_mark.loc[mark_day, mark] = 'S'
            output_dict.update({trading_day:mc_mark})
        return output_dict

    elif catog == 'bm':
        for trading_day in position_dict:
            bm_mark = pd.DataFrame().reindex_like(position_dict[trading_day])
            position_df = position_dict[trading_day].copy().T
            # Total number of days used for collecting data before each trading days
            amount_days = len(position_df.columns)
            # Total number of tickers
            amount_tickers = len(position_df.index)
            for day in range(amount_days):
                bm_daily = position_df.iloc[:,day].copy()
                bm_daily.sort_values(inplace=True)
                high = bm_daily[-int(amount_tickers*0.3):].index.values
                low = bm_daily[:int(amount_tickers*0.3)].index.values
                medium = bm_daily[int(amount_tickers*0.3):-int(amount_tickers*0.3)].index.values
                mark_day = bm_mark.index[day]
                for mark in high:
                    if mark in bm_mark.columns:
                        bm_mark.loc[mark_day, mark] = 'H'
                for mark in low:
                    if mark in bm_mark.columns:
                        bm_mark.loc[mark_day, mark] = 'L'
                for mark in medium:
                    if mark in bm_mark.columns:
                        bm_mark.loc[mark_day, mark] = 'M'
            output_dict.update({trading_day:bm_mark})
        return output_dict

def get_factor(ret_dict, market_dict, mc_dict, bm_dict, intersection=False):
    """
    Calculate factor in each trading period, based on each trading days.
    ----------------------------------------------------------------------------------------------------------------
    Compared with traditional Fama-French method, the method I used in this function is a simplified one.
    Instead of getting Intersection of ('B','S') and ('H', 'M', 'L'), I simply split market capital to 'Big' and
    'Small', then use mc weighted average return with 'Small' tag 'minus' mc weighted average return with tag 'Big'
    to get SMB. Similar procedure to get HML.
    ----------------------------------------------------------------------------------------------------------------
    :return: Save different df(Rm, SMB, HML) in a dict, based on trading days (key).
    """
    if intersection is True:
        return get_factor_intersect(ret_dict, market_dict, mc_dict, bm_dict)

    output_dict = {}
    mc_mark = get_factor_mark(mc_dict, 'mc')
    bm_mark = get_factor_mark(bm_dict, 'bm')

    for traday in ret_dict:     # Data records among 252 days before trading day
        days = len(ret_dict[traday].index)
        tickers = len(ret_dict[traday].columns)
        SMB = []
        HML = []
        for day in range(days):     # Each day
            # Small Minus Big
            mark_small = [ticker for ticker in range(tickers) if mc_mark[traday].iloc[day,ticker] == 'S']
            S = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, mark_small)
            mark_big = [ticker for ticker in range(tickers) if mc_mark[traday].iloc[day,ticker] == 'B']
            B = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, mark_big)
            SMB.append(S - B)

            # High Minus Low
            mark_high = [ticker for ticker in range(tickers) if bm_mark[traday].iloc[day,ticker] == 'H']
            H = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, mark_high)
            mark_low = [ticker for ticker in range(tickers) if bm_mark[traday].iloc[day, ticker] == 'L']
            L = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, mark_low)
            HML.append(H - L)

        # Rm = rm - rf
        Rm = market_dict[traday]['log_Rm']
        factor = pd.DataFrame({'log_Rm':Rm, 'SMB':SMB, 'HML':HML})
        output_dict.update({traday:factor})

    return output_dict

def get_factor_intersect(ret_dict, market_dict, mc_dict, bm_dict):
    """
    Calculate factor in each trading period, based on each trading days.
    """
    output_dict = {}
    mc_mark = get_factor_mark(mc_dict, 'mc')
    bm_mark = get_factor_mark(bm_dict, 'bm')

    for traday in ret_dict:
        days = len(ret_dict[traday].index)
        tickers = len(ret_dict[traday].columns)
        SMB = []
        HML = []
        for day in range(days):     # Each day

            SH_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'S') and (bm_mark[traday].iloc[day,ticker] == 'H')
            ]

            SM_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'S') and (bm_mark[traday].iloc[day,ticker] == 'M')
            ]

            SL_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'S') and (bm_mark[traday].iloc[day,ticker] == 'L')
            ]

            SH = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, SH_mark)
            SM = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, SM_mark)
            SL = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, SL_mark)

            BH_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'B') and (bm_mark[traday].iloc[day,ticker] == 'H')
            ]

            BM_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'B') and (bm_mark[traday].iloc[day,ticker] == 'M')
            ]

            BL_mark = [
                ticker for ticker in range(tickers)
                if (mc_mark[traday].iloc[day,ticker] == 'B') and (bm_mark[traday].iloc[day,ticker] == 'L')
            ]

            BH = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, BH_mark)
            BM = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, BM_mark)
            BL = _factor_weight_average(ret_dict[traday], mc_dict[traday], day, BL_mark)

            # Small Minus Big
            SMB.append((SH + SM + SL)/3 - (BH + BM + BL)/3)

            # High Minus Low
            HML.append((BH + SH)/2 - (BL + SL)/2)

        # Rm = rm - rf
        Rm = market_dict[traday]['log_Rm']
        factor = pd.DataFrame({'log_Rm': Rm, 'SMB': SMB, 'HML': HML})
        output_dict.update({traday: factor})

    return output_dict


def get_alpha(ret_dict, factor_dict, market_dict):
    """
    Calculate and save alpha and its corresponding p-value into a dict, based on trading days.
    """
    output_dict = {}
    for traday in ret_dict:
        tickers = len(ret_dict[traday].columns)
        alpha = []
        p_value = []
        for ticker in range(tickers):
            ticker_ret = ret_dict[traday].iloc[:, ticker] - market_dict[traday]['daily_rf']
            ticker_ret.name = ret_dict[traday].iloc[:, ticker].name[0]
            df = pd.concat([ticker_ret, factor_dict[traday]], axis=1)
            formula = ticker_ret.name + ' ~ log_Rm + SMB + HML'
            ols = sm.OLS.from_formula(formula, df).fit()
            alpha.append(ols.params[0])
            p_value.append(ols.pvalues[0])
        score = pd.DataFrame({'alpha': alpha, 'p_value': p_value}, index=ret_dict[traday].columns.get_level_values(0))
        score.sort_values(by=['alpha'], inplace=True)
        output_dict.update({traday:score})
    return output_dict

def _factor_weight_average(ret_df, mc_df=None, day=None, mark_index=None):
    if len(mark_index) == 0:
        return 0
    else:
        ret = ret_df.iloc[day, mark_index]
        mc = mc_df.iloc[day, mark_index]
        weight = mc / mc.sum()
        res = np.average(ret, weights=weight)
        return res

def __position_check(df, daterange):
    for i in df.index.values:
        if i not in daterange.astype('datetime64[ns]'):
            df.drop([i], inplace=True)
    return df

def __date_position(input_date, target):
    for pos, value in enumerate(input_date):
        if str(value) == target:
            return pos


# ---------------------------------------------------------------------
def main(top:int = 5):
    # Get sp500 data and risk free (annualized percentage) rate
    market = get_market(db)
    # Base on time range of market data, date when we initialize our position and gap between each trading days,
    # get time range and position range.
    date_range = market['date'].values.copy()
    # date_range_64 = market['date'].copy().astype('datetime64')
    # date_position = date_range[252::21]
    # date_position_64 = date_range_64[252::21]
    # Add daily return and risk free (decimal) into market df
    market_position = get_ret(market, date_range[1:], 'S&P500')

    # Get ticker list
    ticker_list = get_ticker(db)
    # Get close price
    close = get_close(ticker_list, date_range, db)
    # Get daily ret and split them based on trading days
    ret = get_ret(close, date_range, 'close', True)
    ret_position = position_split(ret, date_range[1:])
    # Get daily market capital and split them based on trading days
    mc = get_marketcap(ticker_list, date_range, db)
    mc_position = position_split(mc, date_range[1:])
    # Get daily bm ratio, split them based on trading days and mark them based on High, Medium and Low
    bm = get_bm(ticker_list, date_range, db)
    bm_position = position_split(bm, date_range[1:])

    # Get 3 factors: Rm, SMB and HML. Save them into dict, based on trading days.
    factor_position = get_factor(ret_position, market_position, mc_position, bm_position, intersection=True)

    # Calculate alpha for all tickers in each trading period.
    alpha = get_alpha(ret_position, factor_position, market_position)

    # Input all results from above procedures into class Alpha
    # close = __position_check(close, date_range)
    alpha_strategy = pf.Alpha(principle=100000, pool=ticker_list, close=close, alpha=alpha, market=market,
                              date_range=date_range[1:], top=top)
    alpha_strategy.backtest()

    return alpha_strategy


#%%
if __name__ == '__main__':
    alpha_5 = main()

#%%
alpha_15 = Alpha(principle=100000, pool=alpha_5.pool, close=alpha_5.close, alpha=alpha_5.alpha,
                market=alpha_5.market, date_range=alpha_5.date_range, top=15)
alpha_15.backtest()
