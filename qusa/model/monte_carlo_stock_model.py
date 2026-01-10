# Imports
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm 
import yfinance as yf
from datetime import date

# %%
# Import data
def inputs(ticker, start_date):

    today = date.today()
    data_df = yf.download(ticker, start = start_date, end = today)

    return data_df

# %% 
# Data processing
def preprocessing_data(data_df):
    """
    This function performs basic preprocessing of the input data. 
    
    """

    final_data_df = data_df.reset_index()
    final_data_df = final_data_df['Close']

    return final_data_df

# %%
# Calculate log returns
def log_returns(df):
    """
    This function takes the dataframe and calculates the log returns 

    """

    log_return = (np.log(1+df.pct_change()))

    return log_return

# %%
# Calculate drift
def drift_calc(df):
    """
    This function takes the df and calculates the drift of the stock

    """
    lr = log_returns(df)
    u = lr.mean()
    var = lr.var()
    drift = u-(0.5*var)

    try:
        return drift.values
    except:
        return drift
    
# %% 
# Calculate daily returns
def daily_returns(df, days, iterations):
    """
    This function takes the df and uses brownian motion to calculate the daily returns. 
    The inputs are the df, days, and iterations. 
    Days are the number of days that you want to forecast for. 
    Iterations are the number of times you want to run the simulation. 
    
    """
    ft = drift_calc(df)
    try:
        stv = log_returns(df).std().values
    except:
        stv = log_returns(df).std()
    dr = np.exp(ft + stv * norm.ppf(np.random.rand(days, iterations)))

    return dr

# %%
# Probability will return higher than x value 
def probs_find(predicted, higherthan, on = 'value'):
    if on == 'return': 
        predicted0 = predicted.iloc[0,0]
        predicted = predicted.iloc[-1]
        predList =list(predicted)
        over = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 >= higherthan]
        less = [(i*100)/predicted0 for i in predList if ((i-predicted0)*100)/predicted0 < higherthan]
    elif on == 'value':
        predicted = predicted.iloc[-1]
        predList = list(predicted)
        over = [i for i in predList if i >= higherthan]
        less = [i for i in predList if i < higherthan]
    else:
        print( "'on' must be either value or return")
    return (len(over)/(len(over)+len(less)))


# %%
# Monte Carlo simulation full run
def simulate_mc(ticker, start_date, days, iterations, plot=True):
    """
    This function performs the monte carlo simulation, creates a few plots of the distributions and 
    outputs the likelihood of obtaining a particular stock price. 

    """
    data_df = inputs(ticker, start_date)

    final_data_df = preprocessing_data(data_df)

    log_return = log_returns(final_data_df)

    drift = drift_calc(final_data_df)

    dr = daily_returns(final_data_df, days, iterations)


    # Generate daily returns
    returns = daily_returns(final_data_df, days, iterations)
    # Create empty matrix
    price_list = np.zeros_like(returns)
    # Put the last actual price in the first row of the matrix
    price_list[0] = final_data_df.iloc[-1]
    # Calculate the price of each day 
    for t in range(1, days):
        price_list[t] = price_list[t-1]*returns[t]

    # Plot Option
    if plot == True:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1,2, figsize=(14,4))
        sns.distplot(x, ax=ax[0])
        sns.distplot(x, hist_kws={'cumulative':True}, kde_kws={'cumulative':True}, ax=ax[1])
        plt.xlabel("Stock Price")
        plt.show()


    # Plotting final_data_df
    plt.plot(final_data_df.index[-20:], final_data_df.values[-20:], label='SPY')

    # Getting the index range for price_list starting after the last index value of final_data_df
    price_list_index = range(len(final_data_df), len(final_data_df) + len(price_list))

    # Plotting price_list
    plt.plot(price_list_index, price_list, label='Price List')

    # Adding legend and labels
    # plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Price')

    # Displaying the plot
    plt.show()


    # Printing info about stock
    try:
        [print(nam) for nam in final_data_df.columns]
    except:
        print(final_data_df.name)

    print(f"Days: {days}")
    print(f"Exected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(),2)}")
    print(f"Return: {round(100*(pd.DataFrame(price_list).iloc[-1].mean()-price_list[0,1])/pd.DataFrame(price_list).iloc[-1].mean(),2)}%")
    print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list),0, on='return')}")
    print(f"Latest Price: ", final_data_df.iloc[-1,])
    print(f"Quantile (0.5%): ", np.percentile(price_list, 0.5))
    print(f"Quantile (1%): ", np.percentile(price_list, 1))
    print(f"Quantile (5%): ", np.percentile(price_list, 5))
    print(f"Quantile (10%): ", np.percentile(price_list, 10))
    print(f"Quantile (25%): ", np.percentile(price_list, 25))
    print(f"Quantile (50%): ", np.percentile(price_list, 50))
    print(f"Quantile (75%): ", np.percentile(price_list, 75))
    print(f"Quantile (95%): ", np.percentile(price_list, 95))

    return pd.DataFrame(price_list)


# %% 


simulate_mc("AMZN", "2016-10-31", 1, 1000)
simulate_mc("AMZN", "2016-10-31", 3, 1000)
simulate_mc("AMZN", "2016-10-31", 7, 1000)
simulate_mc("AMZN", "2016-10-31", 14, 1000)
simulate_mc("AMZN", "2016-10-31", 30, 1000)
simulate_mc("AMZN", "2016-10-31", 35, 1000)
simulate_mc("AMZN", "2016-10-31", 40, 1000)
simulate_mc("AMZN", "2016-10-31", 45, 1000)


# %%
def backtest_once(price_series, horizon, iterations):
    """
    Use data up to the second-last-horizon to simulate the horizon and compare to actual.
    Returns a dictionary with mean_pred, rmse, and coverage_5_95.
    """
    split = -horizon
    train = price_series.iloc[:split]  # Use .iloc for positional slicing
    actual = price_series.iloc[split + horizon - 1]  # Use .iloc for positional indexing
    actual = actual.item()  # Convert to scalar value
    sim = daily_returns(train, horizon, iterations)  # Simulated daily returns
    
    # Build end prices from simulations
    last_price = train.iloc[-1]
    if isinstance(last_price, pd.Series):  # Ensure it's a scalar
        last_price = last_price.values[0]
    end_prices = last_price * np.prod(sim, axis=0)  # 1D array of final prices for each simulation
    
    # Calculate metrics
    mean_pred = end_prices.mean()
    rmse = np.sqrt(np.mean((end_prices - actual) ** 2))
    mape = np.mean(np.abs((end_prices - actual) / actual)) * 100
    lower, upper = np.percentile(end_prices, 5), np.percentile(end_prices, 95)
    coverage = 1.0 if (actual >= lower and actual <= upper) else 0.0

    return {"mean_pred": mean_pred, "rmse": rmse, "mape": mape, "coverage_5_95": coverage}

# Example backtest for different horizons
ticker = "AMZN"
start_date = "2016-10-31"
data_df = inputs(ticker, start_date)
final_data_df = preprocessing_data(data_df)

horizons = [1, 3, 7, 14, 30]
iterations = 1000
results = []

for horizon in horizons:
    result = backtest_once(final_data_df, horizon, iterations)
    result["horizon"] = horizon
    results.append(result)

# Print results
for res in results:
    print(f"Horizon: {res['horizon']} days")
    print(f"Mean Predicted Price: {res['mean_pred']}")
    print(f"RMSE: {res['rmse']}")
    print(f"MAPE: {res['mape']:.2f}%")
    print(f"Coverage (5%-95%): {res['coverage_5_95']}")
    print("-" * 30)
# %%
