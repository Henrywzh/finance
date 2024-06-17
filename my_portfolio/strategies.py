import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('~/PycharmProjects/algo_trading/finance/my_portfolio/my_portfolio.csv')

price_df = pd.pivot(df, values='Close', index='Date', columns='Ticker')
vol_df = pd.pivot(df, values='Volume', index='Date', columns='Ticker')


# def calcualte_ma(price_list, period: int):
#     if isinstance(price_list, pd.Series):
#         return price_list.rolling(period).mean()
#     elif isinstance(price_list, list):
#         return


def fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None):
    # _df contains the Close of a stock
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'MA{fast}'] = _df[ticker_name].rolling(fast).mean()
    _df[f'MA{slow}'] = _df[ticker_name].rolling(slow).mean()

    _df['Signal'] = np.where(
        _df[f'MA{slow}'].isna(),
        0,
        np.where(_df[f'MA{slow}'] < _df[f'MA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    # return a dataframe with columns: Price, Strategy Return, Cumulative Return
    # it also contains other stuff, but less important
    return _df


# need to redo the ema functions
def ema_fast_slow(df_prev: pd.DataFrame, fast: int, slow: int, ticker_name: str = None):
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()

    _df['Signal'] = np.where(
        _df[f'EMA{slow}'].isna(),
        0,
        np.where(_df[f'EMA{slow}'] < _df[f'EMA{fast}'], 1, -1)
    )  # fill na with 0, if fast MA > slow MA, signal = 1, else -1
    _df['Position'] = _df['Signal'].shift(1)
    _df['Strategy Return'] = _df[ticker_name].pct_change() * _df['Position']
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    return _df


def buy_and_hold(df_prev: pd.DataFrame, ticker_name: str = None):
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df['Strategy Return'] = _df[ticker_name].pct_change()
    _df['Cumulative Return'] = (1 + _df['Strategy Return']).cumprod()

    return _df


def bollinger_bands(df_prev: pd.DataFrame, period: int, step: float, ticker_name: str = None):
    if period < 1: raise ValueError('Period must be >= 1')
    if step <= 0: raise ValueError('Step must be > 0')

    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'MA{period}'] = _df[ticker_name].rolling(period).mean()
    _df['Upper Band'] = _df[f'MA{period}'] + step * _df[f'MA{period}'].rolling(period).std()
    _df['Lower Band'] = _df[f'MA{period}'] - step * _df[f'MA{period}'].rolling(period).std()


# default period: 14
def rsi(df_prev: pd.DataFrame, period: int = 14, ticker_name: str = None):
    if period < 1: raise ValueError('Period must be >= 1')
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df['Diff'] = _df[ticker_name].diff()
    _df['Gain'] = np.where(_df['Diff'] > 0, _df['Diff'], 0)
    _df['Loss'] = np.where(_df['Diff'] < 0, -_df['Diff'], 0)

    _df['Avg Gain'] = np.nan
    _df['Avg Loss'] = np.nan
    first_avg_gain = _df['Gain'][:period].mean()
    first_avg_loss = _df['Loss'][:period].mean()
    _df['Avg Gain'][period - 1] = first_avg_gain
    _df['Avg Loss'][period - 1] = first_avg_loss
    for i in range(period, _df.shape[0]):
        _df['Avg Gain'][i] = ((period - 1) * _df['Avg Gain'][i - 1] + _df['Gain'][i]) / period
        _df['Avg Loss'][i] = ((period - 1) * _df['Avg Loss'][i - 1] + _df['Loss'][i]) / period

    _df['RS'] = _df['Avg Gain'] / _df['Avg Loss']
    _df['RSI'] = 100 - 100 / (1 + _df['RS'])

    return _df


def rsi_2(df_prev: pd.DataFrame, period: int = 14, ticker_name: str = None):
    if period < 1: raise ValueError('Period must be >= 1')
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df['Diff'] = _df[ticker_name].diff()
    _df['Gain'] = np.where(_df['Diff'] > 0, _df['Diff'], 0)
    _df['Loss'] = np.where(_df['Diff'] < 0, -_df['Diff'], 0)

    _df['Avg Gain'] = _df['Gain'].rolling(period).mean()
    _df['Avg Loss'] = _df['Loss'].rolling(period).mean()

    _df['RS'] = _df['Avg Gain'] / _df['Avg Loss']
    _df['RSI'] = 100 - 100 / (1 + _df['RS'])

    return _df


def macd(df_prev: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, ticker_name: str = None):
    if fast < 0: raise ValueError('Fast must be greater than 0')
    if slow < 0: raise ValueError('Slow must be greater than 0')
    if signal < 0: raise ValueError('Signal must be greater than 0')
    if fast > slow:
        temp = fast
        fast = slow
        slow = temp
    if ticker_name is None:
        ticker_name = 'Close'
    _df = pd.DataFrame(df_prev[ticker_name])

    _df[f'EMA{fast}'] = _df[ticker_name].ewm(span=fast, adjust=False).mean()
    _df[f'EMA{slow}'] = _df[ticker_name].ewm(span=slow, adjust=False).mean()
    _df[f'MACD'] = _df[f'EMA{fast}'] - _df[f'EMA{slow}']
    _df[f'Signal Line'] = _df[f'MACD'].ewm(span=signal, adjust=False).mean()

    return _df


def plot_macd(df_prev: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, ticker_name: str = None):
    _df = macd(df_prev, fast, slow, signal, ticker_name=ticker_name)

    plt.figure(figsize=(12, 6))
    x = pd.to_datetime(_df.index.values)

    plt.subplot(2, 1, 1)
    plt.plot(x, _df[ticker_name], label=ticker_name)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, _df['MACD'], label='MACD')
    plt.plot(x, _df['Signal Line'], label='Signal Line')
    plt.bar(x, _df['MACD'] - _df['Signal Line'], label='MACD - Signal')
    plt.legend()
    plt.show()






strategies = {}
FAST = 5
SLOW = 20
ticker_x: str = price_df.columns.values[0]
x_MA = fast_slow(price_df, fast=FAST, slow=SLOW, ticker_name=ticker_x)
x_EMA = ema_fast_slow(price_df, fast=FAST, slow=SLOW, ticker_name=ticker_x)
x_buy_hold = buy_and_hold(price_df, ticker_name=ticker_x)

x_rsi = rsi(price_df, ticker_name=ticker_x)
x_rsi_2 = rsi_2(price_df, ticker_name=ticker_x)

x_macd = macd(price_df, ticker_name=ticker_x)
plot_macd(price_df, ticker_name=ticker_x)


# plt.plot(pd.to_datetime(x_macd.index.values[600:]), x_macd['MACD'][600:], label='MACD')
# plt.plot(pd.to_datetime(x_macd.index.values[600:]), x_macd['Signal Line'][600:], label='Signal')
# plt.legend()
# plt.show()

# strategies['MA Fast and Slow'] = x_MA
# strategies['EMA Fast and Slow'] = x_EMA
# strategies['Buy and Hold'] = x_buy_hold

# for s in strategies:
#     plt.plot(pd.to_datetime(strategies[s].index.values), strategies[s]['Strategy Return'], label=s)
#
# plt.legend()
# plt.xlabel('Date')
# plt.ylabel('Strategy Return')
# plt.show()
