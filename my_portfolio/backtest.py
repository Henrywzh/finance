from finance_utils.strategies import *
from finance_utils.display import *

df = pd.read_csv('./csv_folder/my_portfolio2.csv')
price_df = pd.pivot(df, values='Close', index='Date', columns='Ticker')
vol_df = pd.pivot(df, values='Volume', index='Date', columns='Ticker')

strategies = {}
FAST = 5
SLOW = 20
ticker_x: str = price_df.columns.values[0]
# x_MA = fast_slow(price_df, fast=FAST, slow=SLOW, ticker_name=ticker_x)
# x_EMA = ema_fast_slow(price_df, fast=FAST, slow=SLOW, ticker_name=ticker_x)
# x_buy_hold = buy_and_hold(price_df, ticker_name=ticker_x)
#
# x_rsi = rsi(price_df, ticker_name=ticker_x)
# x_rsi_2 = rsi_2(price_df, ticker_name=ticker_x)
#
# x_macd = macd(price_df, ticker_name=ticker_x)
# plot_macd(price_df, ticker_name=ticker_x)

# x_fast_oscillator = stochastic_oscillator(price_df, ticker_name=ticker_x, type='slow')
# plot_oscillator(price_df, ticker_name=ticker_x)

x_mfi = mfi(df, ticker_x)
plot_mfi(df, ticker_x)

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