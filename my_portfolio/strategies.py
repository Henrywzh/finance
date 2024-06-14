import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

df = pd.read_csv('my_portfolio.csv')

price_df = pd.pivot(df, values='Close', index='Date', columns='Ticker')
vol_df = pd.pivot(df, values='Volume', index='Date', columns='Ticker')

print(price_df, vol_df)
