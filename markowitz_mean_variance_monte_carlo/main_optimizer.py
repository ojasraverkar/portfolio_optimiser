# main_optimizer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def run_portop():
    
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'LICI.NS', 'M&M.NS']

    start_date = '2024-01-01'
    end_date = '2024-12-31'

    try:
        close_df = yf.download(tickers, start=start_date, end=end_date)['Close']        #downloading data for respective tickers
        if close_df.empty:
            print("!!! No data downloaded. Check tickers and dates.")
            return
        close_df.dropna(inplace=True)
    except Exception as e:
        print(f"!!! An error has occured during the data download {e}")
        return
    
    print("Data download completed successfully. !!!")
    #print(close_df.head)

    #log returns are preferred in financial analysis for their time-additive property

    log_returns = np.log(close_df / close_df.shift(1))          
    #pandas fn that pushes all the data down by 1 row. On today's row, the price from yesterday will be placed.
    #then divide today's price by yesterday's price 
    log_returns.dropna(inplace=True)
    #the first row will be empty because there is no previous day. so we drop it here.
    #why log- they are additive overtime. 
    log_returns.to_csv('log_returns.csv', index=False)
    #from this we need to calculate, expected returns, risk and co-movement.
    
    annualized_mean_returns = log_returns.mean() * 252      #avg daily reurn for each stock
    annualized_cov_matrix = log_returns.cov() * 252     #on diagonal we have variances, others are covariance values of pairs of stocks
    #annualized by multiplying by 252.

    #monte-carlooo, randomnly try thousands of combinations to reach an almost accurate optimal result
    num_portfolios = 200000
    num_assets = len(tickers)
    portfolio_returns = np.zeros(num_portfolios)
    portfolio_volatility = np.zeros(num_portfolios)
    portfolio_weights = np.zeros((num_portfolios, num_assets))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_returns[i] = np.sum(weights*annualized_mean_returns)  
        portfolio_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(annualized_cov_matrix, weights)))
        portfolio_weights[i] = weights

    print("Simulation Complete !!!")

    #now we identify the optimal portfolio using sharp ratio which measures the risk-adjusted return
    #assumed risk-free rate of interest is 6%
    risk_free_rate = 0.06

    sharpe_ratios = (portfolio_returns - risk_free_rate)/portfolio_volatility
    #we find the portfolio that has the max of all sahrpe ratios
    max_sharpe_index = np.argmax(sharpe_ratios)
    max_sharpe_returns = portfolio_returns[max_sharpe_index]
    max_sharpe_volatility = portfolio_volatility[max_sharpe_index]
    max_sharpe_weights = portfolio_weights[max_sharpe_index]

    #find the portfolio with lowest risk
    min_vola_index = np.argmin(portfolio_volatility)
    min_vola_return = portfolio_returns[min_vola_index]
    min_vola_volatility = portfolio_volatility[min_vola_index]
    min_vola_weights = portfolio_weights[min_vola_index, :]

    #we will not create a scatter plot for all simulated portfolios
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.7)
    plt.scatter(max_sharpe_volatility, max_sharpe_returns, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')
    plt.scatter(min_vola_volatility, min_vola_return, marker='P', color='b', s=200, label='Min Volatility Portfolio')

    plt.title('Efficient Frontier: Portfolio Optimization', fontsize=18)
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Return', fontsize=12)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)

    print("MONTE CARLO RESULTS READY !!!")
    
    print("\nOptimal Portfolio (Maximum Sharpe Ratio)")
    print(f"Annualized Return: {max_sharpe_returns:.2%}")
    print(f"Annualized Volatility: {max_sharpe_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_index]:.2f}")
    print("\nOptimal Weights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {max_sharpe_weights[i]:.2%}")

    print("\n--- Minimum Volatility Portfolio ---")
    print(f"Annualized Return: {min_vola_return:.2%}")
    print(f"Annualized Volatility: {min_vola_volatility:.2%}")
    print("\nWeights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {min_vola_weights[i]:.2%}")
        
    plt.show()


run_portop()

