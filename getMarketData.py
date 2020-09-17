from pandas_datareader import DataReader
import pymysql
import pandas as pd
import numpy as np

__author__ = 'Han (Aaron) Xiao'

"""
Collect market data for Fama-French Three Factor model.
Get daily index data (S&P500 close price) from yahoo finance and daily risk free data (1-month T-bill rate) from Fred.
Get data from 2015-06-30 to 2019-06-30.
Save data to local SQL database.
"""

start = '2015-06-30'
end = '2019-06-30'

# Get data and
index = DataReader(name='^GSPC', data_source='yahoo', start=start, end=end)
index.dropna(inplace=True)
sp500 = index['Close']
sp500.name = 'sp500_close'
rf = DataReader(name='DGS1MO', data_source='fred', start=start, end=end)
rf.rename(columns={'DGS1MO':'risk_free'}, inplace=True)
rf.dropna(inplace=True)
# Note: here the unit of rf is still the percentage(%) and is annualized


def format_date(df1, df2):
    """
    concat two df into same time range (festival excluded). Fill NA with last valid observation.
    :return:
    """
    data = pd.concat([df1, df2], axis=1)
    data.fillna(method='ffill', inplace=True)
    data = data.reset_index()
    data.iloc[:,0] = data.iloc[:,0].astype(str)

    return to_mysql(data)


def to_mysql(data):
    db = pymysql.connect(host='localhost', user='root', password="",
                         database='ff_3factor', port=3308,
                         charset='utf8')
    cur = db.cursor()

    cur.connection.encoders[np.float64] = lambda value, encoders: float(value)

    sql = 'INSERT INTO `marketdata`(`date`, `S&P500`, `risk_free`) VALUES (%s, %f, %f)'
    for i in range(len(data)):
        cur.execute(sql,(data.iloc[i,0], data.iloc[i,1], data.iloc[i,2]))

    db.commit()
    cur.close()
    db.close()


def main():
    format_date(sp500, rf)
    print('uploaded successfully')


if __name__ == '__main__':
    main()

