# Fama-French-3-Factor-Model-Implementation

## Workflow
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

   ### e.g Top 25 alpha: Portfolio results
   
   | Portfolio Return | Sharpe | Volatility |  IR  | Max Drawdown |
   |  ----  | ----  | ----  | ----  | ----  |    
   | 38.95% | 0.68 | 0.17 |  0.19  | 21.4% |

   The detailed info about trading rules and Alpha Class could be found in portfolio.py
