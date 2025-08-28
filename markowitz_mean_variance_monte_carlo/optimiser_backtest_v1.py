# main_optimizer.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

def backtest_portfolio(weights, tickers, backtest_start, backtest_end, initial_investment=100000):
    """
    Backtests a given portfolio allocation over a specified period and compares it to a benchmark.

    Args:
        weights (np.array): The weights of the assets in the portfolio.
        tickers (list): The list of stock tickers.
        backtest_start (str): The start date for the backtest (e.g., '2024-01-01').
        backtest_end (str): The end date for the backtest (e.g., '2024-12-31').
        initial_investment (float): The starting capital for the simulation.
    """
    print("\n--- Starting Backtest ---")
    print(f"Period: {backtest_start} to {backtest_end}")
    print(f"Initial Investment: ₹{initial_investment:,.2f}")

    # --- Step 1: Download Backtesting Data ---
    # Download data for the portfolio stocks for the backtest period.
    try:
        backtest_prices = yf.download(tickers, start=backtest_start, end=backtest_end)['Close']
        if backtest_prices.empty:
            print("Error: No data downloaded for backtesting period. Check dates.")
            return
        backtest_prices.dropna(inplace=True)
        
        # Download benchmark data (NIFTY 50) for comparison.
        benchmark_prices = yf.download('^NSEI', start=backtest_start, end=backtest_end)['Close']
        if benchmark_prices.empty:
            print("Error: No benchmark data downloaded.")
            return

    except Exception as e:
        print(f"An error occurred during backtest data download: {e}")
        return

    # --- Step 2: Simulate the Investment ---
    # Calculate the number of shares to buy on the first day based on initial investment and weights.
    initial_prices = backtest_prices.iloc[0]
    shares = (initial_investment * weights) / initial_prices
    
    # Calculate the daily value of the portfolio by multiplying shares by daily prices.
    portfolio_value = (backtest_prices * shares).sum(axis=1)

    # --- Step 3: Analyze Performance ---
    # Calculate portfolio total return.
    ending_value = portfolio_value.iloc[-1]
    total_return_portfolio = (ending_value - initial_investment) / initial_investment

    # Calculate benchmark performance (normalized to the same initial investment).
    benchmark_normalized = (benchmark_prices / benchmark_prices.iloc[0]) * initial_investment
    
    # CORRECTED LINE: Explicitly cast the final value to a float to prevent formatting errors.
    ending_benchmark_value = benchmark_normalized.iloc[-1]
    total_return_benchmark = (float(ending_benchmark_value) - initial_investment) / initial_investment

    # Calculate Maximum Drawdown for the portfolio.
    peak = portfolio_value.expanding(min_periods=1).max()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min()

    print("\n--- Backtest Results ---")
    print(f"Ending Portfolio Value: ₹{ending_value:,.2f}")
    print(f"Total Return (Portfolio): {total_return_portfolio:.2%}")
    print(f"Total Return (NIFTY 50): {total_return_benchmark:.2%}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # --- Step 4: Visualize Backtest ---
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot portfolio value over time.
    portfolio_value.plot(label='Optimal Portfolio', color='blue')
    
    # Plot benchmark value over time.
    benchmark_normalized.plot(label='NIFTY 50 Benchmark', color='grey', linestyle='--')
    
    plt.title('Portfolio Backtest Performance vs. NIFTY 50', fontsize=18)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (₹)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.show()


def run_portfolio_optimization(tickers, start_date, end_date, backtest_start_date, backtest_end_date, initial_capital):
    """
    This function performs the entire Markowitz Mean-Variance Portfolio Optimization process.
    It downloads stock data, calculates returns and covariance, runs a Monte Carlo simulation
    to find optimal portfolio weights, and visualizes the results.
    """
    # --- Step 1: Data Acquisition ---
    print(f"Downloading optimization data for: {', '.join(tickers)}...")
    print(f"Optimization Period: {start_date} to {end_date}")
    try:
        adj_close_df = yf.download(tickers, start=start_date, end=end_date)['Close']
        if adj_close_df.empty:
            print("Error: No data downloaded. Check tickers and date range.")
            return
        adj_close_df.dropna(inplace=True)
    except Exception as e:
        print(f"An error occurred during data download: {e}")
        return
    print("Data download complete.")

    # --- Step 2: Calculate Daily Log Returns ---
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns.dropna(inplace=True)

    # --- Step 3: Calculate Annualized Mean Returns and Covariance Matrix ---
    mean_log_returns = log_returns.mean()
    annualized_mean_returns = mean_log_returns * 252
    cov_matrix = log_returns.cov()
    annualized_cov_matrix = cov_matrix * 252

    # --- Step 4: Monte Carlo Simulation ---
    num_portfolios = 20000
    num_assets = len(tickers)
    portfolio_returns = np.zeros(num_portfolios)
    portfolio_volatility = np.zeros(num_portfolios)
    portfolio_weights = np.zeros((num_portfolios, num_assets))
    
    print(f"\nRunning Monte Carlo simulation for {num_portfolios} portfolios...")
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_weights[i, :] = weights
        portfolio_returns[i] = np.sum(weights * annualized_mean_returns)
        portfolio_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(annualized_cov_matrix, weights)))
    print("Simulation complete.")

    # --- Step 6: Identify the Optimal Portfolio (Maximum Sharpe Ratio) ---
    risk_free_rate = 0.072 
    sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_volatility
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    max_sharpe_volatility = portfolio_volatility[max_sharpe_idx]
    max_sharpe_weights = portfolio_weights[max_sharpe_idx, :]
    min_vol_idx = np.argmin(portfolio_volatility)
    min_vol_return = portfolio_returns[min_vol_idx]
    min_vol_volatility = portfolio_volatility[min_vol_idx]
    min_vol_weights = portfolio_weights[min_vol_idx, :]

    # --- Step 5: Visualize the Efficient Frontier ---
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.7)
    plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')
    plt.scatter(min_vol_volatility, min_vol_return, marker='P', color='b', s=200, label='Min Volatility Portfolio')
    plt.title('Efficient Frontier: Portfolio Optimization', fontsize=18)
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Return', fontsize=12)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Sharpe Ratio', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    
    # --- Display Results ---
    print("\n--- Optimal Portfolio (Maximum Sharpe Ratio) ---")
    print(f"Annualized Return: {max_sharpe_return:.2%}")
    print(f"Annualized Volatility: {max_sharpe_volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratios[max_sharpe_idx]:.2f}")
    print("\nOptimal Weights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {max_sharpe_weights[i]:.2%}")

    print("\n--- Minimum Volatility Portfolio ---")
    print(f"Annualized Return: {min_vol_return:.2%}")
    print(f"Annualized Volatility: {min_vol_volatility:.2%}")
    print("\nWeights:")
    for i, ticker in enumerate(tickers):
        print(f"  {ticker}: {min_vol_weights[i]:.2%}")
        
    plt.show()

    # --- RUN THE BACKTEST ON THE OPTIMAL PORTFOLIO ---
    backtest_portfolio(max_sharpe_weights, tickers, backtest_start_date, backtest_end_date, initial_capital)


if __name__ == '__main__':
    # --- SCRIPT CONFIGURATION PANEL ---
    # Define the stocks you want to analyze.
    STOCKS = ['RELIANCE.NS', 'ASHOKLEY.NS', 'COALINDIA.NS', 'HCLTECH.NS', 'INFY.NS', 'IRCTC.NS', 'TATAMOTORS.NS', 'TECHM.NS' ]
    
    # Define the period for portfolio optimization (training data).
    OPTIMIZATION_START_DATE = '2025-01-01'
    OPTIMIZATION_END_DATE = date.today().strftime('%Y-%m-%d')
    
    # Define the period for backtesting (out-of-sample data).
    BACKTEST_START_DATE = '2025-01-01'
    BACKTEST_END_DATE = date.today().strftime('%Y-%m-%d')  #date.today().strftime('%Y-%m-%d') # Automatically set to today's date
    
    # Define the initial investment amount for the backtest.
    INITIAL_CAPITAL = 1000000

    # Run the full analysis.
    run_portfolio_optimization(
        tickers=STOCKS,
        start_date=OPTIMIZATION_START_DATE,
        end_date=OPTIMIZATION_END_DATE,
        backtest_start_date=BACKTEST_START_DATE,
        backtest_end_date=BACKTEST_END_DATE,
        initial_capital=INITIAL_CAPITAL
    )
