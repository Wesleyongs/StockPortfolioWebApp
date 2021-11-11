# %% [markdown]
# ### Buy/Hold/Sell Signals Function

# %%
import pandas as pd
import numpy as np
from pandas_datareader import data
import datetime
from datetime import date

from ta.trend import ema_indicator, MACD
from ta.momentum import rsi
from ta.volume import on_balance_volume, volume_price_trend

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# %%
def get_trade_signal(ticker, price_mvmt, trd_days, signal_date=date.today().strftime("%Y-%m-%d")):
    #####################################################################
    ## 1. Import price data for ticker ##################################
    #####################################################################
    
    if datetime.datetime.now().time() < datetime.time(21, 0, 0):
        signal_date=(date.today()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = '1990-01-01'
    end_date = date.today().strftime("%Y-%m-%d")
    df = data.DataReader(ticker, 'yahoo', start_date, end_date)
    
    #####################################################################
    ## 2. Create trading signal based on price_mvmt and trd_days ########
    #####################################################################
    
    # create conditions
    df['Close_1M_Ltr'] = df['Close'].shift(-trd_days)
    conditions = [
        df['Close_1M_Ltr'].isnull(),
        df['Close_1M_Ltr'] > df['Close']*(1+price_mvmt),
        df['Close_1M_Ltr'] < df['Close']*(1-price_mvmt)
    ]
    values = [None, 'buy', 'sell']

    # create signal
    df['Signal'] = np.select(conditions, values, default='hold')
    df['Signal'] = df['Signal'].astype('category')
    df.drop(columns='Close_1M_Ltr', inplace=True)
    
    #####################################################################
    ## 3. Create features ###############################################
    #####################################################################
    
    # 3a. Moving averages 
    # define short and long time periods for moving averages
    short_term = [5, 10, 20]
    long_term = [50, 100, 200]
    
    # calculate SMA
    for days in (short_term + long_term):
        col = 'MA' + str(days)
        df[col] = df['Close'].rolling(window=days).mean()

    # calculate EMA
    for days in (short_term + long_term):
        col = 'EMA' + str(days)
        df[col] = ema_indicator(df['Close'], window=days)
        
    # calculate SMA/EMA relative to close price
    for ma in df.filter(regex='MA').columns:
        col = ma + '_rel_to_Close'
        df[col] = (df['Close'] - df[ma]) / df['Close']
    
    # calculate short SMA relative to long SMA
    for i in short_term:
        for j in long_term:
            col = 'MA' + str(i) + '_rel_to_MA' + str(j)
            short = df['MA' + str(i)]
            long = df['MA' + str(j)]
            df[col] = (short - long) / long
        
    # calculate short EMA relative to long EMA
    for i in short_term:
        for j in long_term:
            col = 'EMA' + str(i) + '_rel_to_EMA' + str(j)
            short = df['EMA' + str(i)]
            long = df['EMA' + str(j)]
            df[col] = (short - long) / long
            
    # calculate SMA crossovers
    for i in short_term:
        for j in long_term:
            col = 'MA' + str(i) + '_cross_MA' + str(j)
            short = df['MA' + str(i)]
            long = df['MA' + str(j)]

            conditions = [
                (short.shift(1).isnull()) | (long.shift(1).isnull()),
                (short > long) & (short.shift(1) < long.shift(1)),
                (short < long) & (short.shift(1) > long.shift(1)),
            ]
            values = [None, 1, -1]
            df[col] = np.select(conditions, values, default=0)
            df[col] = df[col].astype('category')
        
    # calculate EMA crossovers
    for i in short_term:
        for j in long_term:
            col = 'EMA' + str(i) + '_cross_EMA' + str(j)
            short = df['EMA' + str(i)]
            long = df['EMA' + str(j)]

            conditions = [
                (short.shift(1).isnull()) | (long.shift(1).isnull()),
                (short > long) & (short.shift(1) < long.shift(1)),
                (short < long) & (short.shift(1) > long.shift(1)),
            ]
            values = [None, 1, -1]
            df[col] = np.select(conditions, values, default=0)
            df[col] = df[col].astype('category')
            
    # 3b. RSI and MACD
    # calculate RSI based on standard 14-day window and longer-term 20-day window 
    df['RSI14'] = rsi(close=df['Close'], window=14)
    df['RSI20'] = rsi(close=df['Close'], window=20)
    
    # calculate MACD based on standard settings
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_Diff'] = macd.macd_diff()
    
    # 3c. Volume Indicators
    df['OBV'] = on_balance_volume(close=df['Close'], volume=df['Volume'])
    df['VPT'] = volume_price_trend(close=df['Close'], volume=df['Volume'])
    
    # 3d. Month
    df['Month'] = df.index.month.astype('category')
    
    #####################################################################
    ## 4. Subset Variables ##############################################
    #####################################################################

    variables = df.copy().filter(regex='Signal|rel|cross|RSI|MACD_Diff|Volume|OBV|VPT|Month')
    variables.dropna(inplace=True)
    
    #####################################################################
    ## 5. Randomforest model ############################################
    #####################################################################
    
    # split train and test datasets
    X = variables.loc[:, variables.columns != 'Signal']
    y = variables.loc[:, 'Signal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    
    # fit tuned model
    rf = RandomForestClassifier(max_depth=35, n_estimators=1500, random_state=0)
    rf.fit(X_train, y_train)
    
    # generate prediction on test
    y_test_pred = rf.predict(X_test)
    
    # plot confusion matrix
    matrix = confusion_matrix(y_test, y_test_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    
    fig, ax1 = plt.subplots(figsize=(16,9))
    sns.heatmap(matrix, annot=True, annot_kws={'size':10},
                cmap=plt.cm.Greens, linewidths=0.2, ax=ax1)
    class_names = ['Buy', 'Hold', 'Sell']
    tick_marks = np.arange(len(class_names)) + 0.5
    tick_marks2 = tick_marks
    plt.xticks(tick_marks, class_names, rotation=0)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted Value', figure=fig)
    plt.ylabel('Actual Value')
    plt.title('Confusion Matrix for Random Forest Model for ' + ticker)
    
    # evaluation metrics on test
    eval_metrics = classification_report(y_test, y_test_pred)
    
    #####################################################################
    ## 5. Generate signal ###############################################
    #####################################################################
    
    # generate predictions on whole df
    df_out = df.copy().filter(regex='rel|cross|RSI|MACD_Diff|Volume|OBV|VPT|Month')
    df_out.dropna(inplace=True)
    df_out['Pred'] = rf.predict(df_out)
    df_out = pd.concat([df_out, df['Signal']], axis=1)
    
    # extract signal for chosen date
    if signal_date not in df_out.index:
        trd_signal = 'Hold'
    else:
        trd_signal = df_out.loc[signal_date].at['Pred']
    
    return trd_signal, fig, eval_metrics

# %%
signal, cfsn_matrix, metrics = get_trade_signal(ticker='TSLA', price_mvmt=0.07, trd_days=21, signal_date='2021-11-05')

# %%
print(signal)
print(metrics)

# %%
cfsn_matrix

# %%
if __name__ == '__main__':
    pass


