import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import datetime

start_date = datetime.date(2023, 6, 1)
end_date = datetime.date(2024, 6, 1)

df = yf.download('GBPHKD=X', start=start_date, end=end_date)

print(df)

