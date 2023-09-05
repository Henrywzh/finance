"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           Inflation, %   R-squared:                       0.830
Model:                            OLS   Adj. R-squared:                  0.828
Method:                 Least Squares   F-statistic:                     323.8
Date:                Tue, 05 Sep 2023   Prob (F-statistic):          1.06e-100
Time:                        22:48:57   Log-Likelihood:                -309.47
No. Observations:                 270   AIC:                             628.9
Df Residuals:                     265   BIC:                             646.9
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const          -45.1689      1.905    -23.714      0.000     -48.919     -41.419
SCPI             0.4801      0.059      8.175      0.000       0.364       0.596
NROU, %          8.4282      0.360     23.393      0.000       7.719       9.138
JOLT, k          0.0009   3.58e-05     25.972      0.000       0.001       0.001
Oil Price, $     0.0386      0.002     19.268      0.000       0.035       0.043
==============================================================================
Omnibus:                        2.842   Durbin-Watson:                   0.433
Prob(Omnibus):                  0.241   Jarque-Bera (JB):                3.008
Skew:                          -0.057   Prob(JB):                        0.222
Kurtosis:                       3.504   Cond. No.                     2.36e+05
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.36e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# https://www.newyorkfed.org/research/policy/gscpi#/interactive
# https://fred.stlouisfed.org/series/UNRATE
# https://fred.stlouisfed.org/series/NROU#0
# https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RWTC&f=M
# https://fred.stlouisfed.org/series/JTSJOL
# https://fred.stlouisfed.org/series/FEDFUNDS
# https://fred.stlouisfed.org/series/GDP
# https://www.usinflationcalculator.com/inflation/historical-inflation-rates/

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

quarter_df = pd.read_excel('/Users/wuzihao/Desktop/csvFile/NROU.xls')
month_df = pd.read_excel('/Users/wuzihao/Desktop/csvFile/usInflationData.xlsx')
quarter_df = quarter_df.dropna()
month_df = month_df.dropna()

quarter_to_month_df = quarter_df.loc[quarter_df.index.repeat(3)].reset_index(drop=True)
quarter_to_month_df = quarter_to_month_df[32:]

# print(quarter_to_month_df.head(50))
# print(month_df.tail(10))

df = pd.DataFrame(month_df)
df['NROU, %'] = list(quarter_to_month_df['NROU, %'])
df['GDP Growth, %'] = list(quarter_to_month_df['GDP Growth, %'])

# print(df.head(10))
# print(df.tail(10))

# ---- shift 1 for pred

new_df = pd.DataFrame(df['Inflation, %'])
new_df['SCPI'] = df['SCPI'].shift(1)
new_df['UN Rate, %'] = df['UN Rate, %'].shift(1)
new_df['NROU, %'] = df['NROU, %'].shift(1)
new_df['JOLT, k'] = df['JOLT, k'].shift(1)
new_df['Oil Price, $'] = df['Oil Price, $'].shift(1)
new_df['GDP Growth, %'] = df['GDP Growth, %'].shift(1)
new_df['Fed Funds, %'] = df['Fed Funds, %'].shift(1)

new_df['Date'] = df['Date']
new_df.set_index('Date', inplace=True)
print(new_df)

new_df = new_df.dropna()

# ---- Corr
print(new_df.corr())

# ---- Linear Regression
X = new_df[['SCPI', 'NROU, %', 'JOLT, k', 'Oil Price, $']] #, 'UN Rate, %' , 'Fed Funds, %'
Y = new_df['Inflation, %']

model = LinearRegression()
model.fit(X, Y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Calculate R-squared
y_pred = model.predict(X)
r_squared = r2_score(Y, y_pred)

print("R-squared:", r_squared)

X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
#view model summary
print(model.summary())




# plot graph
plt.plot(new_df.index, list(new_df['Inflation, %']), label='Inflation')
plt.plot(new_df.index, y_pred, label='Pred Inflation')
plt.xlabel('Date')
plt.ylabel('Inflation %')
plt.title('US inflation prediction')
plt.legend()
plt.show()

def us_inflation_pred(scpi, nrou, jolt, oil):
    eqn = -45.16893744142226 + scpi*4.80074295e-01 + nrou*8.42820787e+00 + jolt*9.28743470e-04 + oil*3.86391563e-02
    return eqn
