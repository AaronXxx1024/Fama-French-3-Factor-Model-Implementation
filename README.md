# Fama-French-3-Factor-Model-Implementation

## About
A realization of classic Fama French Three Factor Model for the purpose of empirical study.

## Data
S&P500 constituent stocks from 2015-06-30 to 2019-06-30.
Dataset includes their daily colse price, outstanding share, market cap and book-to-market ratio.
[SQL Data](https://drive.google.com/file/d/12UDgK708uDZOyi2JbFBr0L6IqghwqBBX/view?usp=sharing)  

## Workflow
1. Get data (ticker pool, S&P500, risk free rate, close price, market cap and book-to-market ratio) from SQL database.
2. Organize data to the form we want: {trading_day:df_data}
3. Calculate factor in two different ways:  
  * First way is spliting tickers to 'Big'(50%) and 'Small'(50%) based on market cap, then using market cap weighted average return ('Small'-'Big') to get 'SMB' Split tickers to 'High'(30%), 'Medium'(40%) and 'L'(30%) based on book-to-market ratio and using market cap weighted average return ('High'-'Low') to get 'HML'.  

  * The difference for second way is after initial separation, we do one more step. Mark tickers based on intersection, e.g. if aticker is marked as 'Big' and 'High' in the same time, then we mark it as 'B/H'. Therefore, we'll have total 6 different groups: B/H, B/M, B/L, S/H, S/M and S/L.  

  * Finally, use market cap weighted average return with intersection mark to get 'SMB' and 'HML':  

&emsp;    <a href="https://www.codecogs.com/eqnedit.php?latex=SMB&space;=&space;(SH&space;&plus;&space;SM&space;&plus;&space;SL)&space;/&space;3&space;-&space;(BH&space;&plus;&space;BM&space;&plus;&space;BL)&space;/&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SMB&space;=&space;(SH&space;&plus;&space;SM&space;&plus;&space;SL)&space;/&space;3&space;-&space;(BH&space;&plus;&space;BM&space;&plus;&space;BL)&space;/&space;3" title="SMB = (SH + SM + SL) / 3 - (BH + BM + BL) / 3" /></a> 
  
&emsp;    <a href="https://www.codecogs.com/eqnedit.php?latex=HML&space;=&space;(BH&space;&plus;&space;SH)&space;/&space;2&space;-&space;(BL&space;&plus;&space;SL)&space;/&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?HML&space;=&space;(BH&space;&plus;&space;SH)&space;/&space;2&space;-&space;(BL&space;&plus;&space;SL)&space;/&space;2" title="HML = (BH + SH) / 2 - (BL + SL) / 2" /></a>

4. Save all factor data in a df with columns ['Rm', 'SMB', 'HML'].
   where 'Rm' is the log return of S&P500 minus corresponding daily risk free rate.
5. Regress all tickers' log return on factor data, get interception as 'alpha' and its p-value. Save these two data to
   a dict called 'alpha' with form: {trading_day:df_data([alpha, p-value])}.
6. Input all necessary data to a Class to get an 'Alpha' object.
7. Run backtest() method in Alpha object. When backtesting is done, program will plot portfolio cumulative return vs.
   market cumulative return and print the portfolio result in a table form with columns:
   [Portfolio Return | Sharpe | Volatility |  IR  | Max Drawdown].
   
* The detailed info about trading rules and Alpha object could be found in portfolio.py
   
   ### e.g Top 25 alpha: Portfolio results
   
   | Portfolio Return | Sharpe | Volatility |  IR  | Max Drawdown |
   |  :----:  | :----:  | :----:  | :----:  | :----:  |    
   | 38.95% | 0.68 | 0.17 |  0.19  | 21.4% |
    
![image](https://github.com/AaronXxx1024/Fama-French-3-Factor-Model-Implementation/blob/master/Top%2025%20alpha.png)
   
   
