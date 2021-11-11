import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyfolio as pf
import scipy.stats as stat
from pandas_datareader import data
from plotly.offline import init_notebook_mode, iplot
from scipy.stats import norm
from tabulate import tabulate
import datetime as dt


def main(input_df, positions_df):
    
    #prepro
    portfolio_value = 0
    for i in range(len(positions_df)):
        portfolio_value += positions_df['qty'][i]*positions_df['current_prices'][i]
    positions_df['weights'] = [((positions_df['current_prices'][i]*positions_df['qty'][i])/portfolio_value) for i in range(len(positions_df))]
        
    start_date = input_df.date.min()
    end_date = dt.date.today()-dt.timedelta(days=1)
    tickers = positions_df.stock.unique()
    panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
    # panel_data = panel_data.loc['2019-01-01':'2021-11-04']
    panel_data = panel_data.loc[start_date: end_date]
    closes_1y = panel_data[['Close', 'Adj Close']]
    
    # Return Series

    return_series_adj = (closes_1y['Adj Close'].pct_change()+ 1).cumprod() - 1

    return_series_adj.head()
    
    #Portfolio Calculations
    weights = positions_df['weights'].tolist()
    weights
    weighted_return_series_portfolio = weights * (return_series_adj)
    #Sum the weighted returns
    return_series_adj_portfolio = weighted_return_series_portfolio.sum(axis=1)

    #Single Line Plot using Plotly
    sf = return_series_adj_portfolio
    return_series_adj_portfolio_df = pd.DataFrame({'Date':sf.index, 'Weighted Portfolio Returns':sf.values})

    fig1 = px.line(return_series_adj_portfolio_df, x='Date', y='Weighted Portfolio Returns', title='Time Series with Range Slider and Selectors')

    fig1.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig1.update_layout(
        autosize=False,
        width=1500,
        height=500)
   
    #Multi-line plot using Plotly
    
    return_series_adj_portfolio_df = pd.DataFrame({'Date':return_series_adj_portfolio.index, 'Weighted portfolio returns':return_series_adj_portfolio.values})

    for ticker in tickers:
        temp = return_series_adj[ticker].tolist()
        return_series_adj_portfolio_df[ticker] = temp
    
    fig2 = px.line(return_series_adj_portfolio_df, x="Date", y=return_series_adj_portfolio_df.columns,
              hover_data={"Date": "|%B %d, %Y"},
              title='Weighted Portfolio Returns')


    fig2.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig2.update_layout(
            autosize=False,
            width=1500,
            height=500,)
    
    return_series_close = (closes_1y['Close'].pct_change()+ 1).cumprod() - 1 
    weighted_return_series_close_portfolio = weights * (return_series_close)
    return_series_close_portfolio = weighted_return_series_close_portfolio.sum(axis=1)

    ret_portfolio = (1+return_series_adj_portfolio.tail(1))**(1/2)-1

    vol_portfolio = np.sqrt(252) * np.log((return_series_close_portfolio+1)/(return_series_close_portfolio+1).shift(1)).std()

    risk_free_ann_ret_rate = 1.1

    returns_ts = closes_1y['Adj Close'].pct_change().dropna()
    avg_daily_ret = returns_ts.mean()

    returns_ts['RiskFree_Rate'] = risk_free_ann_ret_rate/252
    avg_rf_ret = returns_ts['RiskFree_Rate'].mean()

    for ticker in tickers:
        returns_ts[f"Excess_ret_{ticker}"] = returns_ts[ticker] - returns_ts['RiskFree_Rate']

    weighted_return_portfolio = weights * (returns_ts[tickers])

    returns_portfolio = pd.DataFrame(weighted_return_portfolio.sum(axis = 1), columns = ['wt_portfolio'])

    avg_daily_portfolio = returns_portfolio.mean()

    risk_free_ann_ret_rate1 = 0
    returns_portfolio['RiskFree_Rate'] = risk_free_ann_ret_rate1/252

    avg_rf_ret_portfolio = returns_portfolio['RiskFree_Rate'] .mean()


    returns_portfolio['Excess_ret_portfolio'] = returns_portfolio['wt_portfolio'] - returns_portfolio['RiskFree_Rate']


    sharpe_portfolio = ((avg_daily_portfolio - avg_rf_ret_portfolio) /returns_portfolio['Excess_ret_portfolio'].std())*np.sqrt(252)
    
    sharpe_value = sharpe_portfolio['wt_portfolio']
    
        
    return vol_portfolio, ret_portfolio.reset_index().iloc[0,1], sharpe_value, fig1, fig2

#helper functions
def volatility_check(vol_portfolio, user_input = 0.50):
    if vol_portfolio<=user_input:
        return 'Good'
    else:
        return 'Bad'

def sharpe_check(sharpe_value):
    print(sharpe_value)
    if sharpe_value >= 3:
        return 'Excellent'
    elif sharpe_value >= 2:
        return 'Great'
    elif sharpe_value >= 1:
        return 'Decent'
    else:
        return 'Bad'
    
def ar_check(ret_portfolio, input_ar=2):
    if ret_portfolio > input_ar:
        return 'Passed'
    else:
        return 'Failed'
      
if __name__ == '__main__':
    pass
