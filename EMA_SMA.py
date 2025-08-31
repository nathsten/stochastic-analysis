import numpy as np
import pandas as pd
import pandas_ta as ta
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import TimeFrame
import datetime as dt
import pytz
from time import sleep
from scipy import stats
from matplotlib import pyplot as plt
import statsmodels.api as sm
from time import sleep
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
END_POINT = os.getenv("ALPACA_END_POINT")

api = tradeapi.REST(API_KEY, SECRET_KEY, END_POINT)

cash = 0
position_market_value = 0

# Create the Alpaca API client
api = tradeapi.REST(API_KEY, SECRET_KEY, END_POINT, api_version="v2")

GMT = pytz.timezone("GMT")
trade_active = False

def generate_brownian_motion_params(S_t, ticker:str)->pd.DataFrame:
    time = np.array([i+1 for i in range(len(S_t)-1)])
    S = S_t[1:]
    S_lag = S_t[:-1]
    X = pd.DataFrame({'Time': time, 'S_lag': S_lag})
    X = sm.add_constant(X)
    S_m = sm.OLS(S, X).fit()
    ser = np.sqrt(S_m.mse_resid)
    bm = {
    f'{ticker}:ser': ser,
    f'{ticker}:Time': S_m.params['Time'],
    f'{ticker}:S_lag': S_m.params['S_lag'],
    f'{ticker}:Drift': S_m.params['const'],
    f'{ticker}:Drift_SE': S_m.bse['const'],
    }
    drift_t = [S[i] - time[i]*bm[f'{ticker}:Time'] - S_lag[i]*bm[f'{ticker}:S_lag'] - np.array(S_m.resid)[i] for i in range(len(S))]
    drift_resid = pd.DataFrame({"Drift": drift_t, "Resid": S_m.resid})
    bm[f'{ticker}:Drift_Resid_Corr'] = drift_resid.corr()
    return pd.DataFrame([bm])

def get_stocks(tickers: list[str], timeframe: str, start: str, end: str) -> pd.DataFrame:
    """
    timeframe: 1min | 5min | 15min | 1hour | 1day | 1week
    """
    stock_dfs = []
    
    for ticker in tickers:
        bars = api.get_bars(ticker, timeframe, start, end).df
        bars.reset_index(inplace=True)
        stock_df = pd.DataFrame({
            'timestamp': bars['timestamp'],
            ticker: bars['close']
        })
        stock_dfs.append(stock_df)

    merged_df = stock_dfs[0]
    
    for stock_df in stock_dfs[1:]:
        merged_df = pd.merge(merged_df, stock_df, on='timestamp', how='inner')

    merged_df.reset_index(drop=True, inplace=True)
    
    return merged_df



def trading_logic(tickers: list[str])->None:
    time_now = dt.datetime.now(tz=GMT)
    time_15_min_ago = time_now - dt.timedelta(minutes=16)
    time_30_min_ago = time_now - dt.timedelta(minutes=46)
    global trade_active
    global cash
    global position_market_value
    data = get_stocks(tickers, '1min', time_30_min_ago.isoformat(), time_15_min_ago.isoformat())
    N = 100
    T = 12

    # data.set_index("time", inplace=True)
    
    for ticker in tickers:
    # Calculate EMA and SMA
        data[f'{ticker}:ema_12'] = ta.ema(data[ticker], length=12)
        data[f'{ticker}:sma_12'] = ta.sma(data[ticker], length=12)

        data[f'{ticker}:signal'] = 0
        data[f'{ticker}:signal'][data[f'{ticker}:ema_12'] < data[f'{ticker}:sma_12']] = 1  # Buy
        data[f'{ticker}:signal'][data[f'{ticker}:ema_12'] > data[f'{ticker}:sma_12']] = -1  # Se

    current = data.iloc[-1]

    stock_corr = data[tickers].corr()


    tickers_to_buy = []

    for ticker in tickers:
        if current[f'{ticker}:signal'] == 1:
            tickers_to_buy.append(ticker)

    print(tickers_to_buy)

    if len(tickers_to_buy) == 2:
        # Check active positions
        active_positions = api.list_positions()
        active_tickers = {pos.symbol for pos in active_positions}

        if set(tickers_to_buy) <= active_tickers:
            print("Both positions already active, no action required.")
        else:
            # Run Monte Carlo Simulation for adjustments
            print("Running Monte Carlo Simulation for:", tickers_to_buy)
            price_patterns = {}
            u_t_matrix = np.random.normal(size=(N, T, len(tickers_to_buy)))
            L_u = np.linalg.cholesky(stock_corr)
            for j in range(N):
                for k in range(T):
                    u_t_matrix[j][k] = np.inner(L_u, u_t_matrix[j][k])
            for ticker in tickers:
                data[f'{ticker}:ser'] = 0
                data[f'{ticker}:Time'] = 0
                data[f'{ticker}:S_lag'] = 0
                data[f'{ticker}:Drift'] = 0
                data[f'{ticker}:Drift_SE'] = 0
                data[f'{ticker}:Drift_Resid_Corr'] = None
                for i in range(24, len(data)):
                    S_t = np.array(data[ticker].to_list()[i - 24:i + 1])
                    AR_params = generate_brownian_motion_params(S_t, ticker).iloc[0]
                    data.loc[i, AR_params.index] = AR_params
            
            current = data.iloc[-1]

            for idx, ticker in enumerate(tickers_to_buy):
                price_patterns[ticker] = np.zeros(N)
                for j in range(N):
                    S_t = [0 for _ in range(T)]
                    S_t[0] = current[ticker]
                    for k in range(1, T):
                        drift = np.random.normal(
                            current[f'{ticker}:Drift'], 
                            0.5 * current[f'{ticker}:Drift_SE']
                        )
                        u = current[f'{ticker}:ser'] * u_t_matrix[j][k][idx]
                        L = np.linalg.cholesky(current[f'{ticker}:Drift_Resid_Corr'])
                        dift, u_t = np.dot(L, np.array([drift, u]))
                        S_t[k] = dift + k * current[f'{ticker}:Time'] + \
                            S_t[k - 1] * current[f'{ticker}:S_lag'] + u_t
                    price_patterns[ticker][j] = 100 * (S_t[-1] / S_t[0] - 1)

            price_patterns_df = pd.DataFrame(price_patterns)

                    # Calculate weights based on the Monte Carlo simulation results
            price_patterns_df['Best_Ticker'] = price_patterns_df.idxmax(axis=1)
            counts = price_patterns_df['Best_Ticker'].value_counts()
            weights = {}
            
            # Generate weights for tickers
            for ticker in tickers:
                if ticker in counts:
                    weights[ticker + ':w'] = counts[ticker] / N

            weights_df = pd.DataFrame([weights]).iloc[-1]
            # Calculate the total cash balance (cash + current positions)
            total_cash = cash + position_market_value
            print(f"Total available cash: {total_cash} USD")

            # Allocate cash according to the weights
            for ticker, weight in weights_df.items():
                if weight > 0:
                    allocation = total_cash * weight
                    qty = int(allocation // current[ticker[:-2]])  # Calculate quantity to buy based on allocation
                    
                    # Submit buy orders
                    if qty > 0:
                        api.submit_order(ticker[:-2], qty, "buy", "market", "day")
                        print(f"Bought {qty} shares of {ticker} at {current[ticker[:-2]]} based on weight {weight}")

            print(f"Positions adjusted according to weights: {weights_df.to_dict()}")

            account = api.get_account()
            cash = float(account.cash)
            position_market_value = float(account.portfolio_value) - float(account.cash)
            trade_active = True


    elif len(tickers_to_buy) == 1:
        stock = tickers_to_buy[0]
        active_positions = api.list_positions()
        active_tickers = {pos.symbol for pos in active_positions}

        sell_tickers = []
        for ticker in active_tickers:
            if ticker != stock:
                sell_tickers.append(ticker)

        if stock in active_tickers:
            print(f"Position already active for {stock}, no action required.")
            for position in sell_tickers:
                api.submit_order(position.symbol, qty=int(position.qty), side="sell", type="market", time_in_force="day")
                print(f"Liquidated position in {position.symbol}")
            trade_active = True
        else:
            # Liquidate other positions
            for position in active_positions:
                api.submit_order(position.symbol, qty=int(position.qty), side="sell", type="market", time_in_force="day")
                print(f"Liquidated position in {position.symbol}")

            sleep(2)

            account = api.get_account()
            cash = float(account.cash)
            position_market_value = float(account.portfolio_value) - float(account.cash)

            # Take full cash position in the target stock
            qty = cash // current[stock]
            api.submit_order(stock, qty, "buy", "market", "day")
            print(f"Bought {stock} at price {current[stock]}")
            position_market_value = qty * current[stock]
            cash -= position_market_value
            trade_active = True
        
    else:
        if trade_active:
            active_positions = api.list_positions()
            active_tickers = [pos.symbol for pos in active_positions]
            # Liquidate other positions
            for position in active_positions:
                api.submit_order(position.symbol, qty=int(position.qty), side="sell", type="market", time_in_force="day")
                print(f"Liquidated position in {position.symbol}")
            trade_active = False

        else:
            print("We have no position")
            


def EMA_SMA_Portfolio_Trader(counter:int, tickers:list[str]):
    global position_market_value
    global cash
    global trade_active

    if counter == 0:
        account = api.get_account()
        position_market_value = float(account.portfolio_value) - float(account.cash)
        cash = float(account.cash)

        # Check if a trade is active
        if position_market_value > 0:
            trade_active = True

    try:
        clock = api.get_clock()  
        if clock.is_open:
            trading_logic(tickers)
        else: print("Market not open.")
    except:
        print("Trading Logic Failed. Probably outside opening Hours...")

if __name__ == "__main__":
    counter = 0
    tickers = ['AAPL','XOM']
    while True:
        print("Checking Portfolio...")
        EMA_SMA_Portfolio_Trader(counter, tickers)
        counter += 1
        sleep(60)
