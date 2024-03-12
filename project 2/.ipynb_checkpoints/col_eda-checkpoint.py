#!/usr/bin/env python
# coding: utf-8

# # Coles Stocks Price EDA and Functions

# ## Libraries

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [12, 8]


# ## Load data

# In[30]:


col = pd.read_csv('COL.csv', parse_dates=['Date'], index_col='Date')
col.head()


# In[31]:


col.info()


# In[32]:


col.corr()


# In[33]:


col.describe()


# ## DATA EDA

# ### Explore Data "Close" Prices only

# #### Prepare the data

# In[34]:


# Original close data
close_d_org = col[['Close']].copy()


# In[35]:


# Original close data + fill in non-trading days
close_d = close_d_org.resample('D').ffill()


# In[36]:


#weekly close data
close_w = close_d_org.resample('W').last()


# In[37]:


#monthly data
close_m = close_d_org.resample('M').last()


# #### Functions For EDA

# In[59]:


def inspect_df(df):
    print('Inspect DataFrame:\n', df.head()) 
    print('\n')
    
def inspect_info(df):
    print('Inspect info:\n')
    df.info() 
    print('\n')
    
def plot_df(df):
    df.plot()
    plt.title('Data Visualization')
    plt.show()
    print('\n')
    
def decomposition_df(df, seasonal_period):
    decomposition = seasonal_decompose(df, period=seasonal_period)
    decomposition.plot()
    plt.title('Data Decomposition')
    plt.show()
    print('\n')
    
def rolling_mean(df, rolling_window):
    plt.figure(figsize=(14, 7))
    rolling = df.rolling(window=rolling_window).mean().dropna()
    plt.plot(rolling, label='rolling')
    plt.plot(df)
    plt.legend()
    plt.title('Rolling Mean')
    plt.xlabel('Date')
    plt.ylabel('Price')

def plot_acf_pacf(df, lags):
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    plot_acf(df.dropna(), lags=lags, ax=ax[0], zero=False)
    ax[0].set_title('Autocorrelation Function')
    plot_pacf(df.dropna(), lags=lags, ax=ax[1], zero=False)
    ax[1].set_title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    
def plot_returns_distribution(df):
    returns = df.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    sns.histplot(returns, kde=True, bins=50)
    plt.title('Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.show()
    
def eda_df(df, seasonal_period, rolling_window, lags):
    inspect_df(df)
    inspect_info(df)
    plot_df(df)
    decomposition_df(df, seasonal_period)
    rolling_mean(df, rolling_window)
    plot_acf_pacf(df, lags)
    plot_returns_distribution(df)


# ##### Call the eda_df function for different data

# In[60]:


# original data
eda_df(close_d_org, 30, 30, 48)


# ## Data Transforming and Usecase

# ### Explore Data as a Whole

# In[89]:


import mplfinance as mpf

def candlestick_chart(df, period='1M'):
    print('Candlestick Chart')
    df_period = df.resample(period).agg({'Open': 'first', 
                                         'High': 'max', 
                                         'Low': 'min', 
                                         'Close': 'last',
                                        'Volume': 'last'})
    mpf.plot(df_period, type='candle', style='charles', volume=True, show_nontrading=True)


# In[90]:


candlestick_chart(col, period='1M')


# In[40]:


col.drop('Volume', axis=1).plot(linewidth=0.5)


# In[8]:


col['Volume'].plot()


# #### Normaliing data to see more clear trend in volume

# ##### Normalizing data with MinMaxScaler (not helping)

# In[9]:


from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()
col_nm = mmscaler.fit_transform(col) #array
col_nmdf = pd.DataFrame(col_nm, columns=col.columns, index=col.index)
col_nmdf.head()


# In[10]:


col_nmdf.describe()


# In[11]:


col_nmdf.corr()


# In[12]:


col_nmdf['Volume'].plot()


# ##### Z-score Normalization (not helping)

# In[13]:


from sklearn.preprocessing import StandardScaler
sscaler = StandardScaler()
col_ss = sscaler.fit_transform(col)
col_ssdf = pd.DataFrame(col_ss, columns=col.columns, index=col.index)
col_ssdf.head()


# In[14]:


col['Volume'].plot()


# ##### Robust Scaler (not helping)

# In[15]:


from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
col_rs = rscaler.fit_transform(col)
col_rsdf = pd.DataFrame(col, columns=col.columns, index=col.index)
col_rsdf.head()


# In[16]:


col_rsdf['Volume'].plot()


# ##### Winsorizing (not helping, try frequency fo the data next)

# In[17]:


from scipy.stats.mstats import winsorize
#volume_winsorized = winsorize(data['Volume'], limits=[0.05, 0.05])
col_c = col.copy()
volume_ws = winsorize(col['Volume'], limits=[0.05, 0.05])
col_c['volume_ws'] = volume_ws
col_c.head()


# In[18]:


col_c['volume_ws'].plot()


# ## Other Functions

# In[ ]:




