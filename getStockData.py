from pandas_datareader import DataReader
from bs4 import BeautifulSoup
import requests
import pymysql
import pandas as pd
import numpy as np
import simfin as sf  # for fundamental financial data
import finnhub  # for fundamental financial data
import os
import re

__author__ = 'Han (Aaron) Xiao'

"""
Collect stock data for Fama-French Three Factor model.
This file is used for stock pool consist of S&P500 component stocks.

The agenda is:
1) Get daily price data of historical (2015-06-30 to 2019-06-30) S&P500 component stocks from yahoo finance.
   I used historical S&P500 component list by referring https://github.com/fja05680/sp500 
2) Get outstanding share by scraping data from https://www.sharesoutstandinghistory.com and https://ycharts.com
3) Calculate daily market cap for each company from 2015-06-30 to 2019-06-30
4) Get total asset and total liabilities by using 'simfin' package. (https://github.com/SimFin/simfin-tutorials) 
5) Calculate daily Book to Market ratio for each company from 2015-06-30 to 2019-06-30
6) Upload data fo local mySQL database.
"""

start = '2015-06-30'
end = '2019-06-30'

wd = 'D:\PycharmProjects\Learning\Empirical Study of Active Portfolio Management\Alpha\Fama_French_three_factor_model'

os.chdir(wd)

historical_sp500 = pd.read_csv('S&P 500 Historical Components & Changes(08-23-2020).csv')
historical_sp500['date'] = historical_sp500['date'].astype('datetime64')
historical_sp500.set_index(keys='date', drop=False, inplace=True)

# ---------------------------------------------------------------------
# functions
def get_components(historical, time):
    """
    Get sp500 component stocks at the appointed time.
    :return: list of component stocks with their ticker names
    """
    if time in historical['date']:
        tmp = historical[historical['date'] == time]
        return tmp.iloc[0, 1].split(sep=',')
    else:
        print('Warning: date is not in time range!')

def get_unique(historical):
    """
    Get an unique ticker list from 2015-06-30 to 2019-06-30 all listed tickers.
    """
    date = '2015-06-30'
    times = historical['date'].astype(str)
    for i, v in enumerate(times):
        if v == date:
            times = times[i:]

    unique = []

    for time in times:
        tmp = get_components(historical, time)
        unique.extend(tmp)

    ticker_list = pd.Series(unique).unique()

    return ticker_list

def get_close(ticker_list):
    """
    Get close price for all listed tickers from 2015-06-30 to 2019-06-30.
    """
    from pandas_datareader._utils import RemoteDataError

    close = {}
    removed_tickers = []

    for ticker in ticker_list:
        try:
            data = DataReader(ticker, 'yahoo', start, end, retry_count=5)
            ticker_close = data['Close'].copy()
            ticker_close.name = ticker
            close.update({ticker:ticker_close})
        except (KeyError, RemoteDataError) as kr:  # if ticker is already been acquired(/spun off) by other company.
            try:
                removed_tickers.append(ticker)
            except RemoteDataError as r:
                pass

    return close, removed_tickers

def close_update(close, removed_tickers):
    """
    Update close price for tickers. Since massive data query from yahoo finance will be blocked.
    """
    no_more_tickers = ['default']

    while len(no_more_tickers) > 0:
        close2, removed_tickers2 = get_close(removed_tickers)
        close.update(close2)
        removed_tickers = removed_tickers2
        no_more_tickers = close2.keys()

    return close, removed_tickers

def get_outstandingshare(remain_tickers):
    """
    Get outstanding share of each ticker saved in the input list.
    Reference: https://www.sharesoutstandinghistory.com
    """
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/85.0.4183.83 Safari/537.36"}

    outstandingshare = {}
    missing = []

    for ticker in remain_tickers:
        url = 'https://www.sharesoutstandinghistory.com/?symbol=' + ticker
        data = requests.get(url, headers=headers)
        soup = BeautifulSoup(data.text, 'html.parser')

        try:  # https://www.sharesoutstandinghistory.com/ doesn't have share data of some tickers we want
            table = soup.findAll('table', {'style': 'margin-top: 20px'})[0]
            rows = table.findAll('td', {'class': 'tstyle'})
            data_rows = []

            for row in rows:
                data_rows.append(row.get_text().replace(',', ''))

            date = []
            outstanding_share = []

            for i, v in enumerate(data_rows):
                if i % 2 == 0:
                    date.append(v)
                else:
                    outstanding_share.append(v)

            date = pd.Series(date)
            outstanding_share = pd.Series(outstanding_share)
            result = pd.concat([date, outstanding_share], axis=1)

            # change str to float, based on format (K, M, B, T)
            update_share = format_unit(result[1])
            # merge date and share into same dataframe
            ticker_share = pd.DataFrame({'date': result[0], '{}_outstanding_share'.format(ticker): update_share})
            ticker_share['date'] = ticker_share['date'].astype('datetime64')
            ticker_share.set_index('date', inplace=True)

            outstandingshare.update({ticker:ticker_share})

        except (IndexError, ValueError):
            try:
                missing.append(ticker)
            except ValueError:
                pass

    return outstandingshare, missing

def format_unit(input_list):
    """
    e.g change 123.45M to 123450000
    """
    output = []
    num_replace = {'K': 1000, 'M': 1000000, 'B': 1000000000, 'T': 1000000000000}

    for i in input_list:
        if i[-1] in num_replace:
            num = num_replace[i[-1]]
            value_num = i[:-1]
            value_num = float(value_num) * num
            output.append(value_num)

    return output

def format_df(df):
    """
    delete time range we dont need
    :param df: assume df.index is datetime64
    """
    if isinstance(df, pd.DataFrame):
        df = df[df.index >= start]
        df = df[df.index <= end]

    return df


"""
# For all tickers missing from the initial outstanding_share scratching, I check them manually and keep (3) of
# them as the input list to supplement_outstandingshare function, 4 of them need special treatment and the rest of them
# should be dropped (they either be acquired or no longer existed anymore).
# This should be a drawback of model, because a company that you already hold and would be acquired could give you a
# huge benefit (correspondingly, a company that you hold but it goes bankrupt could cause a huge loss.
# But for now, I just ’roughly‘ delete them. 
"""


def supplement_outstandingshare(missing):
    """
    Get outstanding share data for tickers in missing list from alternative website.
    Reference: https://ycharts.com/companies
    """
    missing_update = {}
    still_missing = []

    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/85.0.4183.83 Safari/537.36"}

    if len(missing) != 0:

        for ticker in missing:

            url = 'https://ycharts.com/companies/{}/shares_outstanding'.format(ticker)
            data = requests.get(url, headers=headers)
            soup = BeautifulSoup(data.text, 'html.parser')

            try:
                # left is most recent data
                left = soup.findAll('table', {'class': 'histDataTable'})[0]
                right = soup.findAll('table', {'class': 'histDataTable'})[1]
                rows_left = left.findAll('td')
                rows_right = right.findAll('td')
                data_rows = []

                for row in rows_left:
                    data_rows.append(row.get_text().replace('\n', ''))
                for row in rows_right:
                    data_rows.append(row.get_text().replace('\n', ''))

                date = []
                outstanding_share = []

                for i, v in enumerate(data_rows):
                    if i % 2 == 0:
                        date.append(v)
                    else:
                        v = v.strip()
                        outstanding_share.append(v)

                date = pd.Series(date)
                outstanding_share = pd.Series(outstanding_share)
                result = pd.concat([date, outstanding_share], axis=1)
                result[0] = result[0].astype('datetime64')
                result.sort_values(by=0, inplace=True)

                # change str to float, based on format (K, M, B)
                update_share = format_unit(result[1])
                # merge type_updated date and unit_updated share into same dataframe
                ticker_share = pd.DataFrame({'date': result[0], '{}_outstanding_share'.format(ticker): update_share})
                ticker_share['date'] = ticker_share['date'].astype('datetime64')
                ticker_share.set_index('date', inplace=True)

                missing_update.update({ticker:ticker_share})

            except IndexError:
                still_missing.append(ticker)

    return missing_update, still_missing

def close_share_update(close, share, miss_update, special_update):
    """
    :param close: dict{'A':Series_A, 'B': Series_B, ...}
    :param share: dict{'A':df_A, 'B':df_B, ...}
    :param miss_update: dict of 2 companies' outstanding share ('CMCSA', 'FRC').
    :param special_update: dict of 3 companies' outstanding share ('DISCA', 'DISCK', 'GOOGL').
    :return: 2 final dicts, close and share, that would be used to get market cap of each company from start to end.
    """
    # add share_miss_update and share_special_update into share
    share.update(miss_update)
    share.update(special_update)

    # update order of share based on order of close
    share = _pos(share, close, None, 'dict')

    # final check before output
    if [i for i in close] == [i for i in share]:
        return close, share
    else:
        print('Warning: Please recheck the order!!')

def _gettickerlist(input_list, list_type='single'):
    output_list = []
    if list_type == 'single':
        for i in input_list:
            if type(i) is pd.DataFrame:
                output_list.append(i.columns[0].replace('_outstanding_share', ''))
            elif type(i) is pd.Series:
                # isinstance(close_test[0], pd.Series)
                output_list.append(i.name)
        return output_list
    elif list_type == 'combined':
        for i in input_list:
            if isinstance(i, pd.DataFrame):
                output_list.append(i.columns[0])
        return output_list
    elif list_type == 'book':
        for i in input_list:
            if isinstance(i, pd.DataFrame):
                output_list.append(i.columns[2].replace('_book', ''))
        return output_list

def _pos(target_list, final_list, input_list=None, catago='share'):
    """
    Update target_dict based on the order in final_list. If input_dict is provided, items in input_dict will be added
    into target_dict at first, then update order based on final_list.
    """
    if (catago == 'share') and (input_list is not None):
        # update content
        target_list.extend(input_list)
        # update order
        output_list = []
        for pos, value in enumerate(final_list):
            for i in target_list:
                if i.columns[0].replace('_outstanding_share', '') == value:
                    output_list.insert(pos, i)
        return output_list

    elif catago == 'close':
        output_list = []
        if input_list is not None:
            target_list.extend(input_list)
        for pos, value in enumerate(final_list):
            for i in target_list:
                if i.name == value:
                    output_list.insert(pos, i)
        return output_list

    elif catago == 'list':
        output_list = []
        if isinstance(target_list, list) and isinstance(input_list, list):
            target_list.extend(input_list)
        for pos, value in enumerate(final_list):
            for i in target_list:
                if i == value:
                    output_list.insert(pos, i)
        return output_list

    elif catago == 'dict':
        output_dict = {}
        if isinstance(target_list, dict) and isinstance(final_list, dict):
            order = [i for i in final_list]
            for value in order:
                for item in target_list:
                    if item == value:
                        output_dict.update({item:target_list[item]})
        return output_dict

def length_update(input_dict, catog=None):
    if catog == 'share':
        share_length_check = length_check(input_dict, 20)
        share_length_update, share_length_update_miss = supplement_outstandingshare(share_length_check)
        if len(share_length_update_miss) == 0:
            for i in share_length_update:
                input_dict[i] = share_length_update[i]
            if len(length_check(input_dict, 20)) == 0:
                return input_dict
            else:
                print('Please find another data source for share_length_update!!!')
                return input_dict, share_length_update
        else:
            print('Please find another data source for share_length_update_miss!!!')
            return share_length_update, share_length_update_miss

def get_marketvalue(close_dict, share_dict):
    """
    Get market value for input ticker list via their daily close price and outstanding share
    :param close_dict: {'A':stock_A, 'B':stock_B, ...}
    :param share_dict: {'A':A_outstanding_share, 'B':B_outstanding_share, ...}
    :return: a dict of df with columns [close_price, outstanding_share, market_value]
    """
    res = {}
    if [i for i in close_dict] == [i for i in share_dict]:
        for x, y in zip(close_dict, share_dict):
            union = pd.concat([close_dict[x], share_dict[y]], axis=1)
            union.fillna(method='ffill', inplace=True)
            union['market_value'] = union[x] * union[(share_dict[y].columns[0])]
            union = format_df(union)
            res.update({x:union})
        return res
    else:
        print('Recheck order in close and share!')

def get_book(marketvalue, data_source=None):
    """
    Get fundamental data from simfin, Finnhub.io or ychart.
    simfin:
    https://github.com/SimFin/simfin-tutorials
    Finnhub:
    https://finnhub.io/docs/api
    https://github.com/Finnhub-Stock-API/finnhub-python
    ychart:
    https://ycharts.com/dashboard/
    """
    if data_source == 'simfin':
        books = {}
        books_miss = []

        final_tickers = [i for i in marketvalue]

        sf.set_data_dir(os.getcwd())
        balance = sf.load_balance(variant='quarterly', market='us')
        required_columns = ['Total Assets', 'Total Liabilities', 'Total Equity']

        for ticker in final_tickers:
            try:
                ticker_book = balance.loc[ticker][required_columns]
                ticker_book.rename(columns={'Total Equity': '{}_book'.format(ticker)})
                books.update({ticker:ticker_book})
            except KeyError:
                books_miss.append(ticker)
        return books, books_miss

    elif data_source == 'finnhub':
        api = 'bta56t748v6oo3au8vi0'
        finnhub_client = finnhub.Client(api_key=api)
        books = {}
        books_miss = []

        if isinstance(marketvalue, dict):
            final_tickers = [i for i in marketvalue]
        elif isinstance(marketvalue, list):
            final_tickers = marketvalue.copy()

        for ticker in final_tickers:
            data = finnhub_client.financials_reported(symbol=ticker, freq='quarterly')['data']
            if len(data) != 0:
                books.update({ticker:data})
            else:
                books_miss.append(ticker)
        return books, books_miss

    elif data_source == 'ychart':
        book = {}
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/85.0.4183.83 Safari/537.36"}
        error = {}

        for ticker in marketvalue:
            date_text = []
            asset_text = []
            liabilities_text = []
            equity_text = []
            book_miss = []
            index_error = []
            value_error = []

            try:
                for page in range(1, 3):
                    url = 'https://ycharts.com/financials/{}/balance_sheet/quarterly/'.format(ticker) + str(page)
                    data = requests.get(url, headers=headers)
                    soup = BeautifulSoup(data.text, 'html.parser')
                    if len(soup) != 0:
                        table = soup.findAll('table', {'id': 'report'})[0]
                        # Get date data:
                        date_sub = soup_format(table, 'ychart', 'date')
                        date_text.extend(date_sub)

                        # Get asset data
                        asset_sub = soup_format(table, 'ychart', 'asset')
                        asset_text.extend(asset_sub)

                        # Get liabilities
                        liabilities_sub = soup_format(table, 'ychart', 'liabilities')
                        liabilities_text.extend(liabilities_sub)

                        # Get equity
                        equity_sub = soup_format(table, 'ychart', 'equity')
                        equity_text.extend(equity_sub)

                    else:
                        book_miss.append(ticker)
                        continue
            except IndexError:
                index_error.append(ticker)

            date_text = pd.Series(date_text)
            asset_text = pd.Series(format_unit(asset_text))
            liabilities_text = pd.Series(format_unit(liabilities_text))
            equity_text = pd.Series(format_unit(equity_text))

            try:
                df = pd.concat([date_text, asset_text, liabilities_text, equity_text], axis=1)
                df.rename(columns={0: 'date', 1: 'Total Assets', 2: 'Total Liabilities', 3: 'Total Equity'},
                          inplace=True)
                df['date'] = df['date'].astype('datetime64[ns]')
                df.set_index('date', inplace=True)
                df.sort_index(axis=0, inplace=True)
                book.update({ticker: df})
            except ValueError:
                value_error.append(ticker)

            error.update({'IndexError':index_error, 'ValueError':value_error})

        return book, book_miss, error

def soup_format(result, source=None, content=None):
    if source == 'ychart':
        if content == 'date':
            date = result.findAll('tr', {'class': 'dateRow bold'})
            rows = date[0].findAll('td', {'class': "right dataCol"})
            date_sub = []
            for row in rows:
                date_sub.append(row.get_text())
            return date_sub
        elif content == 'asset':
            asset_sub = _content_format(result, 'Total Assets')
            return asset_sub
        elif content == 'liabilities':
            liabilities_sub = _content_format(result, 'Total Liabilities')
            return liabilities_sub
        elif content == 'equity':
            equity_sub = _content_format(result, 'Shareholders Equity')
            return equity_sub

def _content_format(table, content:str):
    bold = table.findAll('tr', {'class': 'bold'})
    tmp = []
    for i in bold:
        if i.findAll(text=content):
            tmp.append(i)
    tmp = tmp[0].get_text()
    tmp = tmp.strip()
    tmp = tmp.replace('\n', ',')
    tmp = ''.join(tmp.split())
    asset_sub = tmp.split(',,,,')

    for i, v in enumerate(asset_sub):
        asset_sub[i] = v.replace(',', '')
    asset_sub = asset_sub[1:]
    return asset_sub

def _precisionCheck(init_dict, content=None, precision=1000000):
    output_dict = {}
    input_dict = init_dict.copy()
    if content == 'book':
        for i in input_dict:
            input_dict[i]['book_check'] = input_dict[i]['Total Assets'] - input_dict[i]['Total Liabilities'] - \
                  input_dict[i]['Total Equity']
            # precision tolerance: might come from round error
            if -precision < np.mean(input_dict[i]['book_check']) < precision:
                del input_dict[i]['book_check']
            else:
                output_dict.update({i:input_dict[i]})
        return output_dict

def update_finnhub(input_dict):
    output_dict = {}
    for ticker in input_dict:
        date = []
        asset = []
        liabilities = []
        equity = []
        data = input_dict[ticker]

        for i in range(len(data)):
            ticker_date = data[i]['endDate']

            try:
                asset_num = _findIndex(data[i]['report']['bs'], 'Total assets')
                ticker_asset = data[i]['report']['bs'][asset_num]['value']
            except (TypeError, TypeError):
                try:
                    asset_num = _findIndex(data[i]['report']['bs'], 'Total assets', 'else')
                    ticker_asset = data[i]['report']['bs'][asset_num]['value']
                except TypeError:
                    continue

            try:
                liabilities_num = _findIndex(data[i]['report']['bs'], 'Total liabilities')
                ticker_liabilities = data[i]['report']['bs'][liabilities_num]['value']
            except (TypeError, TypeError):
                try:
                    liabilities_num = _findIndex(data[i]['report']['bs'], 'Total liabilities', 'else')
                    ticker_liabilities = data[i]['report']['bs'][liabilities_num]['value']
                except TypeError:
                    continue

            try:
                book_num = _findIndex(data[i]['report']['bs'], 'StockholdersEquity', 'concept')
                ticker_book = data[i]['report']['bs'][book_num]['value']
            except (TypeError, TypeError):
                try:
                    book_num = _findIndex(data[i]['report']['bs'], "Total shareholders' equity")
                    ticker_book = data[i]['report']['bs'][book_num]['value']
                except TypeError:
                    continue

            date.append(ticker_date)
            asset.append(ticker_asset)
            liabilities.append(ticker_liabilities)
            equity.append(ticker_book)

        date = _checkNone(date)
        asset = _checkNone(asset)
        liabilities = _checkNone(liabilities)
        equity = _checkNone(equity)

        df = pd.DataFrame({'date': date, 'Total Assets': asset, 'Total Liabilities': liabilities,
                           'Total Equity': equity})
        df['date'] = df['date'].astype('datetime64[ns]')
        df.set_index('date', inplace=True)
        df.sort_index(axis=0, inplace=True)
        df = format_df(df)

        output_dict.update({ticker:df})

    return output_dict

def book_attempt(market_cap):
    # First attempt from data source: simfin
    # check book data miss
    book, book_miss = get_book(market_cap, 'simfin')
    # check book data length
    book_length = length_check(book, 17)
    # check book data precision
    book_precision = _precisionCheck(book, 'book')

    if len(book_miss) != 0 or len(book_length) != 0 or len(book_precision) != 0:
        book_retry = book_miss.copy()
        book_retry.extend([i for i in book_length])
        book_retry.extend([i for i in book_precision])

    # Adjust tickers with multiple share class.
    multiple_shareclass = {'GOOG': 'GOOGL', 'DISCA': 'DISCK', 'NWS': 'NWSA', 'UA': 'UAA'}
    for key, value in multiple_shareclass.items():
        if (value in book_miss) and (key in book):
            book[value] = book[key]
    for value in multiple_shareclass.values():
        if value in book_retry:
            book_retry.remove(value)

    # Second attempt from data source: Finnhub.io
    # Free Finnhub API has timeout restriction (30 calls/second), so I split this ticker list to two parts.
    book_update, book_update_miss = get_book(book_retry[:int(len(book_retry) / 2)], 'finnhub')
    book_update2, book_update_miss2 = get_book(book_retry[int(len(book_retry) / 2):], 'finnhub')

    # merge miss book data from 2nd attempt
    book_update.update(book_update2)
    book_update_miss.extend(book_update_miss2)
    # format finnhub data
    book_update = update_finnhub(book_update)
    # check book data length
    book_update_length = length_check(book_update, 17)
    # check book data precision
    book_update_precision = _precisionCheck(book_update, content='book', precision=100000)

    if len(book_update_miss) != 0 or len(book_update_length) != 0 or len(book_update_precision) != 0:
        book_retry2 = book_update_miss.copy()
        book_retry2.extend([i for i in book_update_length])
        book_retry2.extend([i for i in book_update_precision])
        book_retry2 = list(set(book_retry2))

    # Third attempt from data source: ychart
    # todo: book data miss and error
    book_ychart, book_ychart_miss, book_ychart_error = get_book(book_retry2, 'ychart')
    # todo: 检查长度过少的ticker
    book_ychart_length = length_check(book_ychart, 17)
    # todo: 检查精度不够的ticker (round error)
    # book_ychart_precision = _precisionCheck(book_ychart, 'book', 1000000)

    # integrate all book data
    book_update = dict_update(book_update, book_retry2, 'delete')
    book_ychart = dict_update(book_ychart, [i for i in book_ychart_length], 'delete')
    dropped_list3 = ['LM', 'DO', 'STI', 'JEF', 'BKR']
    book_ychart = dict_update(book_ychart, dropped_list3, 'delete')
    book_ychart_update = get_book(book_ychart_miss, 'ychart')[0]
    book.update(book_update)
    book.update(book_ychart)
    book.update(book_ychart_update)

    return book

def get_bm(marketcap, books):
    book_to_market = {}
    error = []
    for cap, book in zip(marketcap, books):
        try:
            union = pd.concat([marketcap[cap], books[book]], axis=1)
            union.fillna(method='ffill', inplace=True)
            union['book-to-market'] = union['Total Equity'] / union['market_value']
            union = format_df(union)
            book_to_market.update({cap:union})
        except ValueError:
            error.append(cap)
    return book_to_market, error

def _bookCheck(input_list):
    companies = sf.load_companies(market='us')
    book_confirm = []

    for i in input_list:
        if i in companies.index.values:
            book_confirm.append(i)

    return book_confirm

def _checkNone(input_list):
    res = []
    for i in input_list:
        if isinstance(i, str) and i == 'N/A':
            res.append(None)
        else:
            res.append(i)
    return res

def _findIndex(input_dict, key, title='label'):
    for i, v in enumerate(input_dict):
        if title == 'label':
            if v[title] == key:
                return i

        elif title == 'concept':
            if v[title] == key:
                return i

        elif title == 'else':
            key = key.replace('Total ', '')
            key = key.title()
            title_alt = 'concept'
            if v[title_alt] == key:
                return i

def findItem(input_iter, target=None):
    for i, v in enumerate(input_iter):
        if v == target:
            return i

def length_check(input_dict, target:int):
    """
    Check len of each value in the input dict. Return dict with keys of those values and their length.
    """
    length = {}
    for i in input_dict:
        if len(input_dict[i]) < target:
            length.update({i:len(input_dict[i])})
    return length

def dict_update(input_dict, other, method=None):
    """
    update dict content (add/delete) based on method and standard list
    :param input_dict: initial dict
    :param other: standard list
    :param method: 'add' or 'delete'
    :return:
    """
    if method == 'delete':
        for key in input_dict.copy():
            if key in other:
                input_dict.pop(key)

        return input_dict

    if method == 'update':
        if isinstance(input_dict, dict) and isinstance(other, dict):
            for ticker_update in other:
                input_dict[ticker_update] = other[ticker_update].copy()
        return input_dict

def get_url_name(ticker):
    input_url = 'https://www.macrotrends.net/stocks/charts/{}/apple/revenue'.format(ticker)
    current_url = requests.get(input_url).url
    company_name = re.findall('{}/(.*?)/revenue'.format(ticker), current_url)[0]
    return company_name

def tmp_format(tmp_list, unit=None):
    output_list = []
    for i in tmp_list:
        i = str(i) + unit
        i = i.replace(',','')
        output_list.append(i)
    output_list = format_unit(output_list)
    return output_list

def scripts_find(scripts, target: str):
    for i in range(len(scripts)):
        if len(scripts[i].contents) != 0:
            if target in scripts[i].contents[0]:
                script = scripts[i].contents[0]
                field = script.split('field_name')
    left = []
    right = []
    for pos, value in enumerate(field):
        value = value.strip()
        if target in value:
            content = field[pos]
            number_str = re.findall('div>","(.*?)},{', content)[0]
            number_str = number_str.replace('"', '')
            number_list = number_str.split(',')
            for number in number_list:
                inner_number = number.split(':')
                if inner_number[1] == '':
                    inner_number[1] = None
                left.append(inner_number[0])
                right.append(inner_number[1])
    return left, right

def book_supp(book_miss):
    book_new = {}
    book_no = []
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/85.0.4183.83 Safari/537.36"}
    for ticker in book_miss:
        company_name = get_url_name(ticker)
        url = 'https://www.macrotrends.net/stocks/charts/{}/{}/balance-sheet?freq=Q'.format(ticker, company_name)
        data = requests.get(url, headers=headers)
        soup = BeautifulSoup(data.text, 'html.parser')
        scripts = soup.findAll('script')
        asset_date, asset_num = scripts_find(scripts, 'Total Assets')
        liabilities_date, liabilities_num = scripts_find(scripts, 'Total Liabilities<')
        share_date, share_num = scripts_find(scripts, 'Share Holder Equity')
        df = pd.DataFrame({'date': asset_date, 'Total Assets': asset_num, 'Total Liabilities': liabilities_num,
                           'Total Equity': share_num})
        df.dropna(axis=0, how='any', inplace=True)
        df['date'] = df['date'].astype('datetime64[ns]')
        df.set_index('date', inplace=True)
        df.sort_index(axis=0, inplace=True)
        for i in range(3):
            df.iloc[:,i] = tmp_format(df.iloc[:,i], 'M')
        book_new.update({ticker: df})

    return book_new

def to_mysql(data_dict):
    """
    save final stock data to local SQL data base
    """
    db = pymysql.connect(host='localhost', user='root', password="",
                         database='ff_3factor', port=3308,
                         charset='utf8')
    cur = db.cursor()

    cur.connection.encoders[np.float64] = lambda value, encoders: float(value)

    sql = 'INSERT INTO `stockdata`(`date`, `ticker`, `close`, `outstanding_share`, `market_value`, `total_assets`, ' \
          '`total_liabilities`, `book_value`, `book_to_market`) VALUES ' \
          '(%s,%s,%f,%f,%f,%f,%f,%f,%f)'
    for ticker in data_dict:
        if data_dict[ticker].columns[0] != 'DateTime':
            data_dict[ticker].reset_index(level=0, inplace=True)
            data_dict[ticker] = data_dict[ticker].rename(columns={'index': 'DateTime'})
            data_dict[ticker]['DateTime'] = data_dict[ticker]['DateTime'].astype(str)

        for row in range(len(data_dict[ticker])):
            cur.execute(sql, (data_dict[ticker].iloc[row,0], ticker,
                              data_dict[ticker].iloc[row,1],
                              data_dict[ticker].iloc[row,2], data_dict[ticker].iloc[row,3],
                              data_dict[ticker].iloc[row,4], data_dict[ticker].iloc[row,5],
                              data_dict[ticker].iloc[row,6], data_dict[ticker].iloc[row,7]))
    db.commit()
    cur.close()
    db.close()

#%% ---------------------------------------------------------------------
def main():
    ticker_list = get_unique(historical_sp500)

    # Get close price and removed tickers based on the time range we choose.
    close, removed_tickers = get_close(ticker_list)
    close, removed_tickers = close_update(close, removed_tickers)
    # Delete tickers that don't have enough price history
    # close_length = length_check(close, 1000)
    dropped_list = ['ADT', 'AET', 'ALTR', 'ANDV', 'CA', 'CCE', 'DOW', 'ESRX', 'IR', 'MON', 'NFX', 'PLL', 'PX', 'SCG',
                    'SE', 'SNI', 'TWX', 'NLOK', 'CSRA', 'XL', 'HPE', 'FCPT', 'FTV', 'EVHC', 'BHF', 'LW', 'CTVA', 'PCP',
                    'PCL', 'FOX', 'FOXA', 'AABA', 'COL']
    close = dict_update(close, dropped_list, 'delete')
    remain_tickers = [i for i in close]

    # Get outstanding share and missing tickers based on remain_tickers.
    share, share_missing = get_outstandingshare(remain_tickers)
    # share_length = length_check(share, 20)
    dropped_list2 = ['DD', 'FTR', 'JCI', 'CCEP', 'CPRI', 'J', 'KDP', 'EMC', 'HAR', 'HOT', 'POM', 'GAS']
    share = dict_update(share, dropped_list2, 'delete')
    close = dict_update(close, dropped_list2, 'delete')

    # Get supplement share data
    # Manually check missing tickers to confirm what tickers should be used to get data from alternative source.
    # Special Case for outstanding share (2015-06-30 to 2019-06-30)
    # 1) DISCA(can vote) and DISCK: stock distribution, CA should be more expensive and less num
    #    for DISCA, it accounts for about 0.3 in total outstanding share, DISCK accounts for about 0.68. since I can
    #    only get total outstanding share, so I need this ratio for the exact MV.
    # 2) GOOG and GOOGL
    #    same for google, GOOG is class C share (the one I can get exactly) and GOOGL is class A share. GOOGL usually
    #    accounts for 0.43 of total outstanding share.
    share_miss = ['CMCSA', 'FRC']
    share_special = ['DISCA', 'DISCK', 'GOOGL']

    share_miss_update, still_missing = supplement_outstandingshare(share_miss)
    share_special_update, special_missing = supplement_outstandingshare(share_special)

    special_ratio = [0.3, 0.68, 0.43]
    if len(still_missing) == 0 and len(special_missing) == 0:
        for x, y in zip(special_ratio, share_special_update):
            share_special_update[y] = share_special_update[y] * x

    # Update close and share, make sure they have same length and same order
    close, share = close_share_update(close, share, share_miss_update, share_special_update)

    # Length update: update data length before merge close and share to get market cap
    share = length_update(share, 'share')

    # Calculate daily market capital
    market_cap = get_marketvalue(close, share)

    # Get book data for remain tickers from close and share
    book = book_attempt(market_cap)

    # Update market_cap content and order based on book data
    market_cap = dict_update(market_cap, [i for i in market_cap if i not in book], 'delete')
    book = _pos(book, market_cap, None, 'dict')

    # Calculate book-to-market ratio
    bm = get_bm(market_cap, book)[0]

    # Before data saving, check if NANs exist.
    bmiss = []
    for i in bm:
        if bm[i].isnull().values.any():
            bmiss.append(i)
    if len(bmiss) != 0:
        bmiss_update = book_supp(bmiss)
        book = dict_update(book, bmiss_update, 'update')
        bm = get_bm(market_cap, book)[0]

    # Save data to local mySQL database
    to_mysql(bm)


if __name__ == '__main__':
    main()
