from seaborn.matrix import heatmap
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
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import cufflinks as cf
import seaborn as sns
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()
import datetime as dt
import plotly.graph_objects as go_offline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from correlation_asset_distribution import *
from volatility_sharpe import *

# Heading
st.set_page_config(layout="wide")
st.write(
    """
# Portfolio Analysis
This app generates the **Portfolio Analysis** report for any given period - This report will provides insights and additional functionalities not commonly found on stock brokerages accounts \n
This app was designed for small hedge funds who do not have in house analytical capabilities  \n
This app logic follows the principles of first in first out (FIFO) \n
If you wish to use your own csv, ensure the input **csv** has the following columns:  
> 1. Date
> 2. Stock Ticker 
> 3. Action   
> 4. Qty 
> 5. Price 

Created by [FA G3](https://wesleyongs.com/).
"""
)

# Helper Functions
def get_data(df):
    """
    Takes in the input df and outputs the positions df among other variables
    Will return positions_df, realised_gains, unrealised_gains, portfolio_size, available_cash
    """
    df_positions = df.copy(deep=True)
    sells = (
        df_positions[df_positions["action"] == "SELL"]
        .sort_values(by="date")
        .reset_index()
    )
    exclude_sells = (
        df_positions[df_positions["action"] != "SELL"]
        .sort_values(by="date")
        .reset_index()
    )
    realised_gains = 0

    # Process first in first out
    for idx, row in sells.iterrows():

        stock = row["stock"]
        count = row["qty"]
        sell_price = row["price"]
        while abs(count) > 0:

            # Find first occurance of stock
            first_index = (exclude_sells.stock.values == stock).argmax()

            # Exact amount
            if exclude_sells.iloc[first_index]["qty"] == abs(count):
                exclude_sells.drop([first_index])
                realised_gains += (
                    sell_price - exclude_sells.at[first_index, "price"]
                ) * count
                count = 0
            # Enough to sell
            elif exclude_sells.iloc[first_index]["qty"] > abs(count):
                exclude_sells.at[first_index, "qty"] += count
                realised_gains += (
                    sell_price - exclude_sells.at[first_index, "price"]
                ) * count
                count = 0
            # Not enough
            else:
                exclude_sells.drop([first_index])
                realised_gains += (
                    sell_price - exclude_sells.at[first_index, "price"]
                ) * exclude_sells.at[first_index, "qty"]
                count += exclude_sells.at[first_index, "qty"]

    # Find the current positions
    positions = {}
    buys = exclude_sells[exclude_sells["action"] == "BUY"].sort_values(by="date")
    for idx, row in buys.iterrows():

        stock = row["stock"]
        qty = int(row["qty"])
        price = int(row["price"])

        if stock not in positions.keys():
            positions[stock] = [qty, price]
        else:
            new_qty = qty + positions[stock][0]
            new_price = (
                (qty * price) + (positions[stock][0] * positions[stock][1])
            ) / new_qty
            positions[stock] = [new_qty, new_price]

    positions_df = (
        pd.DataFrame(data=positions, index=["qty", "price"])
        .T.reset_index()
        .rename(columns={"index": "stock"})
    )

    # Total equity
    positions_df["equity"] = positions_df["qty"] * positions_df["price"]

    # Add in current prices
    tickers = list(df.stock.unique())
    tickers.remove("Cash")

    date = dt.date.today() - dt.timedelta(days=1)

    panel_data = data.DataReader(tickers, "yahoo", date, date)

    current_prices = []

    for idx, row in positions_df.iterrows():

        stock = row["stock"]

        price = panel_data["Close"][stock].tail(1)[0]
        current_prices.append(price)

    positions_df["current_prices"] = current_prices

    # Adding in floating profits
    positions_df["P&L"] = (
        (positions_df["current_prices"] - positions_df["price"])
    ) * positions_df["qty"]

    # Realised gains, unrealised, portfolio size, available cash
    unrealised_gains = positions_df["P&L"].sum()
    portfolio_size = (
        df[df["action"] == "Deposit"]["price"].astype("int").sum()
        - df[df["action"] == "Withdraw"]["price"].astype("int").sum()
    )
    available_cash = (
        df[df["action"] == "Deposit"]["price"].astype("int").sum()
        - (positions_df["price"] * positions_df["qty"]).sum()
    )
    positions_df.round(3)

    # Touching up
    positions_df[["price"]] = positions_df[["price"]].round(2)
    positions_df[["current_prices"]] = positions_df[["current_prices"]].round(2)
    positions_df[["P&L"]] = positions_df[["P&L"]].round(0)

    return (
        positions_df,
        realised_gains,
        unrealised_gains,
        portfolio_size,
        available_cash,
    )


def industry_pie(positions_df):

    sector_list = []
    stocks = positions_df["stock"]
    for each in stocks:
        stock_info = yf.Ticker(each)
        if "sector" in stock_info.info:
            sector_list.append(stock_info.info["sector"])
        else:
            sector_list.append("Others")

    df2 = positions_df.assign(sector=sector_list)

    labels = df2["sector"]

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]]
    )
    fig.add_trace(
        go.Pie(labels=labels, values=df2["qty"] * df2["current_prices"], name="Asset"),
        1,
        1,
    )
    fig.add_trace(go.Pie(labels=labels, values=df2["P&L"], name="P&L"), 1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=0.5)
    fig.update_traces(textposition="outside", textinfo="label+value")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        title_text="Industry Breakdown",
        width=1500,
        height=500,
        font=dict(
            size=10,
        ),
        # Add annotations in the center of the donut pies.
        annotations=[
            dict(text="Asset", x=0.19, y=0.5, font_size=20, showarrow=False),
            dict(text="P&L", x=0.80, y=0.5, font_size=20, showarrow=False),
        ],
    )

    return fig


def positions_pie(positions_df):
    labels = positions_df["stock"]

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]]
    )
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=positions_df["qty"] * positions_df["current_prices"],
            name="Asset",
        ),
        1,
        1,
    )
    fig.add_trace(go.Pie(labels=labels, values=positions_df["P&L"], name="P&L"), 1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=0.5)
    fig.update_traces(textposition="outside", textinfo="label+value")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        title_text="Portfolio Breakdown",
        width=1500,
        height=500,
        font=dict(
            size=10,
        ),
        # Add annotations in the center of the donut pies.
        annotations=[
            dict(text="Asset", x=0.19, y=0.5, font_size=20, showarrow=False),
            dict(text="P&L", x=0.80, y=0.5, font_size=20, showarrow=False),
        ],
    )
    return fig


def ahv_chart(df_portfolio):
    tickers = list(df_portfolio["stock"])
    today = dt.date.today()
    year_ago = dt.date.today() - dt.timedelta(days=365)

    start_date = year_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    panel_data = data.DataReader(tickers, "yahoo", start_date, end_date)

    close_price = panel_data["Close"]
    adj_close = panel_data["Adj Close"]
    ahv = np.sqrt(np.log(close_price / close_price.shift(1)).var()) * np.sqrt(252)
    ahv_pct = ahv * 100

    ahv_pct = pd.DataFrame(ahv_pct)
    ahv_pct.rename(columns={0: "Annualized Historical Volatility"}, inplace=True)
    ahv_pct = ahv_pct.rename_axis("Ticker").reset_index()

    return px.bar(
        ahv_pct,
        x="Ticker",
        y="Annualized Historical Volatility",
        title="Annualized Historical Volatility of individual stocks in portfolio",
        color="Ticker",
    )


def pnl_chart(df_input):
    dates = df_input["date"]
    pnl = [0]
    portfolio_dict = {}
    # st.dataframe(df_input['date'])
    # st.dataframe(dates)

    ## The logic is flawed here, it is only assuming 1 action per day (fix later)
    ## The code doesnt handle repeating the same stock (fix later)
    for i in range(len(df_input)):
        if df_input["action"][i] == "BUY":
            portfolio_dict[df_input["stock"][i]] = [
                df_input["qty"][i],
                df_input["price"][i],
            ]
        if df_input["action"][i] != "SELL":
            pnl += [pnl[-1] + 0]
        elif df_input["action"][i] == "SELL":
            pnl += [
                pnl[-1]
                + abs(df_input["qty"][i])
                * (df_input["price"][i] - portfolio_dict[df_input["stock"][i]][1])
            ]
    pnl = pnl[1:]
    daily_pnl = pd.DataFrame(dates)
    daily_pnl["Daily PnL"] = pd.DataFrame(pnl)
    daily_pnl["date"] = pd.to_datetime(daily_pnl["date"], format="%d-%m-%y")
    return px.line(daily_pnl, x="date", y="Daily PnL", title="Daily PnL")


# Upload Files
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
csv = "input.csv"
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko = f'<a href="data:file/csv;base64,{b64}" download="input.csv">Download sample csv file</a>'
st.markdown(linko, unsafe_allow_html=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["date"], dayfirst=True)
    title = "your"
else:
    df = pd.read_csv("files/input.csv", parse_dates=["date"], dayfirst=True)
    title = "a dummy"

# Clean data
df["date"] = df["date"]

# Analytics section
st.write(f"# Here is a breakdown of {title} portfolio \n")

# Show portfolio and sidebar
positions_df, realised_gains, unrealised_gains, portfolio_size, available_cash = get_data(df)
st.sidebar.header('Your Positions')
st.sidebar.dataframe(positions_df)
# st.sidebar.dataframe(positions_df[['stock','equity','qty']].style.format({'qty':'{:.0f}','equity':'{:.0f}'}))
df_temp = df.copy()
df_temp["date"] = df_temp["date"].dt.date
st.write("### This is the input file you gave")
st.dataframe(df_temp)
st.write('### These are your positions')
st.dataframe(positions_df)
# st.dataframe(positions_df.style.format({'price':'{:.2f}','current_prices':'{:.2f}','qty':'{:.0f}','equity':'{:.0f}','P&L':'{:.0f}'}))
download=st.sidebar.button('Download positions file')
if download:
    "Download Started! Please wait a link will appear below for your to download the file"
    csv = positions_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    linko = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    st.sidebar.markdown(linko, unsafe_allow_html=True)

# Show Analytical Plots

# Full plots
# TODO: Remove top line
vol_portfolio, ret_portfolio, sharpe_value, fig1 , fig2 = main(df, positions_df)
# st.write(fig1)
st.write(fig2)

col1,col2 = st.beta_columns((1,1))   

# Line plot
# fig = ahv_chart(positions_df)
# col1.write(fig)
# fig = pnl_chart(df)
# col2.write(fig)

# Pie plotly
fig = positions_pie(positions_df)
st.plotly_chart(fig)
fig = industry_pie(positions_df)
st.write(fig)

# Prescription section
st.write(f"""# Here are some assesments of your portfolio \n""")

#sliders
correlation_threshold = st.slider("Input your threshold correlation value", min_value=0.0, max_value=1.0, value=0.5)
distribution_threshold = st.slider("Input your asset distribution threshold", min_value=0.0, max_value=1.0, value=0.5)

# Correlation
# TODO: We fix the correlation to 0.8, let them choose proportion of stock 
correlation_status, heatmap = correlation(positions_df, correlation_threshold)
if correlation_status=='Passed':st.success(f"""## Correlation Check: **{correlation_status}**""")
elif correlation_status=='Failed':st.error(f"""## Correlation Check: **{correlation_status}**""")
if correlation_status == "Failed":
    my_expander = st.beta_expander(label="Show More")
    my_expander.pyplot(heatmap)

# Distribution
asset_distribution_status, asset_overthreshold, asset_overthreshold_percentage = asset_distribution(positions_df, distribution_threshold)
if asset_distribution_status=='Passed':st.success(f"""## Asset Distribution Check: **{asset_distribution_status}**""")
if asset_distribution_status=='Failed':st.error(f"""## Asset Distribution Check: **{asset_distribution_status}**""")
if asset_distribution_status == "Failed":
    my_expander = st.beta_expander(label="Show More")
    asset_overthreshold_str = ','.join(asset_overthreshold)
    my_expander.write(f"The following assets exceed your threshold {asset_overthreshold_str}")
    my_expander.write(ahv_chart(positions_df))

# Sharpe ratio
sharpe_ratio_status = sharpe_check(sharpe_value)
if sharpe_ratio_status=='Excellent':st.success(f"""## Sharpe Ratio Check: **{sharpe_ratio_status}**""")
elif sharpe_ratio_status=='Great':st.success(f"""## Sharpe Ratio Check: **{sharpe_ratio_status}**""")
elif sharpe_ratio_status=='Decent':st.info(f"""## Sharpe Ratio Check: **{sharpe_ratio_status}**""")
elif sharpe_ratio_status=='Bad':st.error(f"""## Sharpe Ratio Check: **{sharpe_ratio_status}**""")
# sharp_threshold = st.slider("Input your threshold sharpe ratio", min_value=0.0, max_value=4.0, value=float(sharpe_value))
# if sharp_threshold != 1.0:sharpe_ratio_status = sharpe_check(sharpe_value, sharp_threshold)
# if sharpe_ratio_status == "Failed":
my_expander = st.beta_expander(label="Show More")
my_expander.write(f"Your sharpe ratio is {round(sharpe_value,2)}")


# vol Ratio
# TODO: Let them choose the threshold for volatility
volatility_status = volatility_check(vol_portfolio)
if volatility_status=='Good':st.success(f"""## Volatility Check: **{volatility_status}**""")
elif volatility_status=='Bad':st.error(f"""## Volatility Check: **{volatility_status}**""")
# volatility_threshold = st.slider("Your Volatility Performance", min_value=0.0, max_value=1.0, value=float(vol_portfolio))
# my_bar = st.progress(float(vol_portfolio))
my_expander = st.beta_expander(label="Show More")
my_expander.write(f"Your sharpe ratio is {round(vol_portfolio,2)}")

