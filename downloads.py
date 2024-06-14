import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime
plt.style.use('ggplot')

start_date = datetime.date(2021, 6, 1)
end_date = datetime.date(2024, 6, 1)
tickers = ['6608.HK', '0650.HK', '0354.HK']

# daily_data = yf.download(tickers, start=start_date, end=end_date)

tickers_data = []

for t in tickers:
    df = yf.download(t, start=start_date, end=end_date)
    df['Ticker'] = t
    df['Date'] = df.index.values
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.set_index(pd.Index([i for i in range(df.shape[0])]))
    tickers_data.append(df)

combined_data = pd.concat(tickers_data)
combined_data = combined_data.set_index(pd.Index([i for i in range(combined_data.shape[0])]))

pivot_df = combined_data.pivot(index="Date", columns="Ticker", values="Close")

print(combined_data)
# combined_data.to_csv('my_portfolio.csv')
