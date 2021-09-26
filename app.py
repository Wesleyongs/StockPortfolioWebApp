import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import glob
import os
import sys
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly
import plotly.express as px
import xlsxwriter
import base64
from io import BytesIO
import datetime as dt
from pandas_datareader import data

###########
# heading #
###########
st.set_page_config(layout="wide")
st.write("""
# Portfolio Analysis
This app generates the **Portfolio Analysis** report for any given period - This report will provides insights and additional functionalities not commonly found on stock brokerages accounts \n
This app was design with Tiger Brokers statment export feature \n
If you wish to use your own csv, ensure the input **csv** has the following columns:  
> 1. tbd  
> 2. tbd  
> 3. tbd      

Created by [Someone](https://wesleyongs.com/).
""")

################
# Upload Files #
################

# SG
uploaded_file = st.file_uploader('Upload CSV file', type="csv")
    
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file,
                   parse_dates=['month'])
    title = "Your"
else:    
    df = pd.read_csv('input.csv',
                    parse_dates=['date'], dayfirst=True)
    title = "Dummy"

# Download table 
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

# Helper Functions
def get_data(df):
    '''
    Will return positions_df, realised_gains, unrealised_gains, portfolio_size, available_cash
    '''  
    df_positions = df.copy(deep=True)
    sells = df_positions[df_positions['action'] == 'SELL'].sort_values(by='date')
    exclude_sells = df_positions[df_positions['action'] != 'SELL'].sort_values(by='date')
    realised_gains = 0

    # Process first in first out 
    for idx, row in sells.iterrows():

        stock = row['stock']
        count = row['qty']
        sell_price = row['price']
        while abs(count) > 0:

            # Find first occurance of stock
            first_index = (exclude_sells.stock.values == stock).argmax()

            # Exact amount
            if exclude_sells.iloc[first_index]['qty'] == abs(count):
                exclude_sells.drop([first_index])
                realised_gains += (sell_price - exclude_sells.at[first_index,'price']) * count
                count = 0
            # Enough to sell
            elif exclude_sells.iloc[first_index]['qty'] > abs(count):
                exclude_sells.at[first_index,'qty'] += count
                realised_gains += (sell_price - exclude_sells.at[first_index,'price']) * count
                count = 0
            # Not enough
            else:
                exclude_sells.drop([first_index])
                realised_gains += (sell_price - exclude_sells.at[first_index,'price']) * exclude_sells.at[first_index,'qty']
                count += exclude_sells.at[first_index,'qty']

    # Find the current positions
    positions = {}
    buys = exclude_sells[exclude_sells['action'] == 'BUY'].sort_values(by='date')
    for idx,row in buys.iterrows():

        stock = row['stock']
        qty = int(row['qty'])
        price = int(row['price'])

        if stock not in positions.keys():
            positions[stock] = [qty,price]
        else:
            new_qty = qty + positions[stock][0]
            new_price = ((qty*price) + (positions[stock][0]*positions[stock][1])) / new_qty
            positions[stock] = [new_qty,new_price]
            
    positions_df = pd.DataFrame(data=positions, index=['qty','price']).T.reset_index().rename(columns={'index':'stock'})

    # Add in current prices

    tickers = list(df.stock.unique())
    tickers.remove('Cash')

    date = dt.date.today() - dt.timedelta(days=1)

    panel_data = data.DataReader(tickers, 'yahoo', date, date)

    current_prices = []

    for idx, row in positions_df.iterrows():

        stock = row['stock']

        price = panel_data['Close'][stock].tail(1)[0]
        current_prices.append(price)

    positions_df['current_prices'] = current_prices

    # Adding in floating profits
    positions_df['P&L'] = (positions_df['current_prices'] -
                           positions_df['price']) * positions_df['qty']

    # Realised gains, unrealised, portfolio size, available cash
    print(realised_gains)
    unrealised_gains = positions_df['P&L'].sum()
    print(unrealised_gains)
    portfolio_size = df[df['action'] == 'Deposit']['price'].astype(
        'int').sum() - df[df['action'] == 'Withdraw']['price'].astype('int').sum()
    print(portfolio_size)
    available_cash = df[df['action'] == 'Deposit']['price'].astype(
        'int').sum() - (positions_df['price'] * positions_df['qty']).sum()
    print(available_cash)
    positions_df.round(3)
    
    # Touching up 
    positions_df[['price']] = positions_df[['price']].round(2)
    positions_df[['current_prices']] = positions_df[['current_prices']].round(2)
    positions_df[['P&L']] =  positions_df[['P&L']].round(0)
    
    return positions_df, realised_gains, unrealised_gains, portfolio_size, available_cash

positions_df, realised_gains, unrealised_gains, portfolio_size, available_cash = get_data(df)

st.dataframe(positions_df.round(2))
download=st.button('Download positions file')
if download:
    'Download Started! Please wait a link will appear below for your to download the file'
    csv = positions_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.markdown(linko, unsafe_allow_html=True)

y = positions_df['price'] * positions_df['qty']
mylabels = positions_df['stock']

col1,col2 = st.beta_columns((1,1))

fig3 = plt.figure(figsize=(16, 9))
plt.pie(y, labels = mylabels)
col1.write(fig3,use_column_width=True)