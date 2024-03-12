#!/usr/bin/env python
# coding: utf-8

# # Coles Stocks Price EDA and Functions

# ## Libraries

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [12, 8]


# ## Load data

# In[30]:


col = pd.read_csv('COL.csv', parse_dates=['Date'], index_col='Date')



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


# #### Functions For models
def check_stationary(df):
    result = adfuller(df.diff().dropna())
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    print(result)
    
    # Decision based on p-value
    if p_value < 0.05:
        print(f"The p-value {p_value} is less than 0.05. We reject the null hypothesis. The time series is likely stationary.")
    else:
        print(f"The p-value {p_value} is greater than 0.05. We fail to reject the null hypothesis. The time series is likely non-stationary.")

    # Additional decision logic based on ADF statistic and critical values
    print("Based on the ADF Statistic and critical values:")
    if adf_statistic < critical_values['1%']:
        print("- The series is stationary with 99% confidence.")
    elif adf_statistic < critical_values['5%']:
        print("- The series is stationary with 95% confidence.")
    elif adf_statistic < critical_values['10%']:
        print("- The series is stationary with 90% confidence.")
    else:
        print("- The series is not stationary.")

def train_test(df, date):
    test_start_date = date
    train = df[:test_start_date]
    test = df[test_start_date:]
    return train, test


def predictions_plot(df, test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Actual', color='blue')
    plt.plot(test.index, predictions, label='Predicted', color='red')
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()





# ##### Call the eda_df function for different data


import mplfinance as mpf

def candlestick_chart(df, period='1M'):
    print('Candlestick Chart')
    df_period = df.resample(period).agg({'Open': 'first', 
                                         'High': 'max', 
                                         'Low': 'min', 
                                         'Close': 'last',
                                        'Volume': 'last'})
    mpf.plot(df_period, type='candle', style='charles', volume=True, show_nontrading=True)


from sklearn.preprocessing import MinMaxScaler
mmscaler = MinMaxScaler()
col_nm = mmscaler.fit_transform(col) #array
col_nmdf = pd.DataFrame(col_nm, columns=col.columns, index=col.index)


from sklearn.preprocessing import StandardScaler
sscaler = StandardScaler()
col_ss = sscaler.fit_transform(col)
col_ssdf = pd.DataFrame(col_ss, columns=col.columns, index=col.index)


# ##### Robust Scaler (not helping)

# In[15]:


from sklearn.preprocessing import RobustScaler
rscaler = RobustScaler()
col_rs = rscaler.fit_transform(col)
col_rsdf = pd.DataFrame(col, columns=col.columns, index=col.index)




from scipy.stats.mstats import winsorize
#volume_winsorized = winsorize(data['Volume'], limits=[0.05, 0.05])
col_c = col.copy()
volume_ws = winsorize(col['Volume'], limits=[0.05, 0.05])
col_c['volume_ws'] = volume_ws

# model fitting function



def generate_X_y(df, window_size, prediction_day):
    X = []
    y = []
    for i in range(len(df) - window_size - prediction_day + 1):
        X.append(df[i : i + window_size].values)
        y.append(df[i + window_size + prediction_day - 1])
    X = np.array(X)
    y = np.array(y)
    return X, y

def spliting_data(X, y, train_ratio, train_val_ratio):
    train_cut = int(X.shape[0]*train_ratio)
    val_cut = int(X.shape[0]*train_val_ratio)
    X_train, X_val, X_test = X[:train_cut], X[train_cut:val_cut], X[val_cut:]
    y_train, y_val, y_test = y[:train_cut], y[train_cut:val_cut], y[val_cut:]
    return X_train, X_val, X_test, y_train, y_val, y_test


def scaler_option(X_train, X_val, X_test, scaler):
    if scaler == 0:
        return X_train, X_val, X_test
    else:
        datascaler = scaler
        X_train_scaled = datascaler.fit_transform(X_train)
        X_val_scaled = datascaler.transform(X_val)
        X_test_scaled = datascaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

from sklearn.linear_model import LinearRegression
def fit_predict(models, X_train, X_val, y_train):
    model = models
    model.fit(X_train, y_train)
    predict_train = model.predict(X_train)
    predict_val = model.predict(X_val)
    return model, predict_train, predict_val


from sklearn.metrics import mean_squared_error as MSE, r2_score
def evaluate(y, predict):
    error = np.sqrt(MSE(y, predict))
    score = r2_score(y, predict)
    return error, score


def plot_actual_vs_predicted(actual, predicted, title='Actual vs. Predicted'):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def model_training(df, window_size, prediction_day, train_ratio, train_val_ratio, models, scaler, plot):
    X, y = generate_X_y(df, window_size, prediction_day)
    X_train, X_val, X_test, y_train, y_val, y_test = spliting_data(X, y, train_ratio, train_val_ratio)
    X_train_scaled, X_val_scaled, X_test_scaled = scaler_option(X_train, X_val, X_test, scaler)
    model, predict_train, predict_val = fit_predict(models, X_train, X_val, y_train)
    error_train, score_train = evaluate(y_train, predict_train)
    error_val, score_val = evaluate(y_val, predict_val)
    print(f"Training - RMSE: {error_train}, R2: {score_train}")
    print(f"Validation - RMSE: {error_val}, R2: {score_val}")
    if plot == 0:
        print("Change last parameter to none-zero to plot: Actual vs predicted")
    else: 
        plot_actual_vs_predicted(y_train, predict_train, title='Training: Actual vs. Predicted')
        plot_actual_vs_predicted(y_val, predict_val, title='Validation: Actual vs. Predicted')
    
    return model, X_train_scaled, y_train, X_test_scaled, y_test, X_val_scaled, y_val, error_train, score_train, error_val, score_val 


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
def cross_val(model, param_grid, X_train, y_train):    
    # Set up the GridSearchCV to search the parameters
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=tscv, n_iter=100, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=42)
    
    # Fit the GridSearchCV to the data
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    # Best parameters and score
    
    print("Best parameters:", random_search.best_params_)
    print("Best rmse:", np.sqrt(-random_search.best_score_))  # Assuming we're interested in RMSE
    return best_model

def test_result(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)  # Use X_test if no scaling applied
    rmse = np.sqrt(MSE(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("RMSE:", rmse)
    print("R2", r2) 
    return rmse, r2


















