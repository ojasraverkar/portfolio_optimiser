#app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date

def port_op(tickers, start_date, end_date, num_portfolios, risk_free_rate):
    #data downloading
    try:
        with st.spinner(f"Downloading Data for Selected Stocks..."):
            close_df = yf.download(tickers, start=start_date, end=end_date)["Close"]
        if close_df.empty == True:
            st.error("!!! No Data Downloaded. Check tickers and date range. Common issue - Check if the stock existed in the range of selected dates. !!!")
            return None, None, None
        
        close_df.dropna(inplace=True)
        if close_df.empty == True:
            st.warning("Dataframe is Empty After Dropping zeroes. Check you dates and Selected Stocks again")
            return None, None, None
        
    except Exception as e:
        st.error(f"A critical error occured during downloading data - {e}")
        return None, None, None
    
    st.success("Data Download Completed Succcessfully !!!")

    log_returns = np.log(close_df / close_df.shift(1))          
    #pandas fn that pushes all the data down by 1 row. On today's row, the price from yesterday will be placed.
    #then divide today's price by yesterday's price 
    log_returns.dropna(inplace=True)
    #the first row will be empty because there is no previous day. so we drop it here.
    #why log- they are additive overtime. 
    #log_returns.to_csv('log_returns.csv', index=False)
    #from this we need to calculate, expected returns, risk and co-movement.
    
    annualized_mean_returns = log_returns.mean() * 252      #avg daily reurn for each stock
    annualized_cov_matrix = log_returns.cov() * 252     #on diagonal we have variances, others are covariance values of pairs of stocks
    #annualized by multiplying by 252.

    #monte-carlooo, randomnly try thousands of combinations to reach an almost accurate optimal result
    num_assets = len(tickers)
    portfolio_returns = np.zeros(num_portfolios)
    portfolio_volatility = np.zeros(num_portfolios)
    portfolio_weights = np.zeros((num_portfolios, num_assets))

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_returns[i] = np.sum(weights*annualized_mean_returns)  
        portfolio_volatility[i] = np.sqrt(np.dot(weights.T, np.dot(annualized_cov_matrix, weights)))
        portfolio_weights[i] = weights

        progress = (i + 1) / num_portfolios
        progress_bar.progress(progress)
        status_text.text(f"Simulating Portfolios: {i+1}/{num_portfolios}")

    print("Simulation Complete !!!")
    progress_bar.empty()

    sharpe_ratios = (portfolio_returns - risk_free_rate)/portfolio_volatility
    #we find the portfolio that has the max of all sahrpe ratios
    max_sharpe_index = np.argmax(sharpe_ratios)
    max_sharpe_returns = portfolio_returns[max_sharpe_index]
    max_sharpe_volatility = portfolio_volatility[max_sharpe_index]
    max_sharpe_weights = portfolio_weights[max_sharpe_index]
    max_sharpe_ratio = sharpe_ratios[max_sharpe_index]

    max_sharpe_portfolio = {
        "Return": max_sharpe_returns,
        "Volatility": max_sharpe_volatility,
        "Sharpe Ratio": max_sharpe_ratio,
        "Weights": max_sharpe_weights
    }

    #find the portfolio with lowest risk
    min_vola_index = np.argmin(portfolio_volatility)
    min_vola_return = portfolio_returns[min_vola_index]
    min_vola_volatility = portfolio_volatility[min_vola_index]
    min_vola_weights = portfolio_weights[min_vola_index, :]

    min_vol_portfolio = {
        "Return": min_vola_return,
        "Volatility": min_vola_volatility,
        "Weights": min_vola_weights
    }

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.7)
    ax.scatter(max_sharpe_volatility, max_sharpe_returns, marker='*', color='r', s=200, label='Max Sharpe Ratio Portfolio')
    ax.scatter(min_vola_volatility, min_vola_return, marker='P', color='b', s=200, label='Min Volatility Portfolio')

    ax.set_title('Efficient Frontier: Portfolio Optimization', fontsize=18)
    ax.set_xlabel('Annualized Volatility (Risk)', fontsize=12)
    ax.set_ylabel('Annualized Return', fontsize=12)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Sharpe Ratio', fontsize=12)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True)

    return fig, max_sharpe_portfolio, min_vol_portfolio

st.set_page_config(layout='wide')

st.title("Portfolio Optimizer - Powered by Monte.Carlo Method")
st.write("A Portfolio Optimizer is an investment tool. It helps us to define how much weightage we must give to stocks, all while keeping the risk under check.")

#sidebar
st.sidebar.header('User Input Parameters')

#tickers
default_tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'LICI.NS', 'M&M.NS']
tickers_input = st.sidebar.text_area("Enter Tickers (comma-separated)", ', '.join(default_tickers))
tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

#date Range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", date(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", date.today())

#simulation Parameters
num_portfolios = st.sidebar.number_input("Number of Portfolios to Simulate", min_value=1000, max_value=500000, value=100000, step=1000)
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=6.0, step=0.1) / 100

#main_app_logic
if st.sidebar.button('Run Optimization'):
    if not tickers or tickers == ['']:
        st.warning("You forgot to enter the tickers!!")
    else:
        fig, max_sharpe_portfolio, min_vol_portfolio = port_op(tickers, start_date, end_date, num_portfolios, risk_free_rate)

        if fig:
            st.subheader("Efficient Frontier Chart")
            st.pyplot(fig)
            
            st.subheader("Optimal Portfolio Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("Optimal Portfolio (Maximum Sharpe Ratio)")
                if max_sharpe_portfolio:
                    st.write(f"**Annualized Return:** {max_sharpe_portfolio['Return']:.2%}")
                    st.write(f"**Annualized Volatility:** {max_sharpe_portfolio['Volatility']:.2%}")
                    st.write(f"**Sharpe Ratio:** {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
                    
                    weights_df_max_sharpe = pd.DataFrame(max_sharpe_portfolio['Weights'], index=tickers, columns=['Weight'])
                    weights_df_max_sharpe['Weight'] = weights_df_max_sharpe['Weight'].map('{:.2%}'.format)
                    st.dataframe(weights_df_max_sharpe)

            with col2:
                st.info("Minimum Volatility Portfolio")
                if min_vol_portfolio:
                    st.write(f"**Annualized Return:** {min_vol_portfolio['Return']:.2%}")
                    st.write(f"**Annualized Volatility:** {min_vol_portfolio['Volatility']:.2%}")
                    
                    weights_df_min_vol = pd.DataFrame(min_vol_portfolio['Weights'], index=tickers, columns=['Weight'])
                    weights_df_min_vol['Weight'] = weights_df_min_vol['Weight'].map('{:.2%}'.format)
                    st.dataframe(weights_df_min_vol)
else:
    st.info("Adjust the parameters in the sidebar and click 'Run Optimization' to begin.")

