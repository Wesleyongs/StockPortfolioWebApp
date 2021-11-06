#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
from pandas_datareader import data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

# Check1: Correlation between stocks. Outputs fig, check_status
# TODO: Add in parameter so user can choose own threshold
def correlation(positions_df, threshold=0.5):

    #Yahoo finance
    tickers = positions_df['stock'].tolist()
    start_date = '2019-01-01'
    end_date = '2021-11-04'
    panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
    panel_data= panel_data['Close']
    ret_series= (panel_data.pct_change()+1).cumprod()-1
    corr = ret_series.corr()
    #print(corr)
    
    #Graph
    fig, ax = plt.subplots(figsize=(16,9))
    sns.heatmap(corr, annot = True, ax = ax, cmap="YlGnBu")
    
    #Calcualte no of stock that are correlated 
    check = "Passed"
    length = len(tickers) #no of tickers
    new = [] # list of correlation
    for each in corr:
        row = corr[each]
        new.extend(row.tolist())
    total = len(new)-length # total no of pairs
    count = 0 #correlation >0.8
    for i in new:
        if(abs(i) >= threshold):
            count+=1
    count = count-length
    if count/total>=0.5:
        check= "Failed"
        print("More than 50% of the stock that are highly positively correlated")
    else:
        print("Less than 50% of your stock that are highly positively correlated")
    return check, fig


# In[6]:

# Check1: Asset distribution between stocks. Outputs check_status
# TODO: Add in parameter so user can choose own threshold
def asset_distribution(positions_df, user_input=0.5):
    total_assets = positions_df['qty']*positions_df['current_prices'] 
    assets = positions_df['stock'].tolist()
    total = sum(total_assets) #total amount of assets
    assets_percentage = []
    # get the distribution of each asset
    for asset in total_assets:
        assets_percentage.append(asset/total)
    # get asset which over the threshold
    asset_overthreshold = []
    asset_overthreshold_percentage = []
    for i in range(len(assets)):
        if assets_percentage[i] >= user_input:
            asset_overthreshold.append(assets[i])
            asset_overthreshold_percentage.append(assets_percentage[i])
    # show result
    check = "Passed"
    if len(asset_overthreshold)>0:
        check="Failed"
        print("Asset distribution is over the threshold")
        for n in range(len(asset_overthreshold)):
            percentage = asset_overthreshold_percentage[n]*100
            print("{}: {:.1f}%".format(asset_overthreshold[n],percentage))
    else:
        print("Asset distribution is below the threshold")
    return check, asset_overthreshold, asset_overthreshold_percentage
    
if __name__ == '__main__':
    pass