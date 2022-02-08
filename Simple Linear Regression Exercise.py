import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

data = pd.read_csv("real_estate_price_size.csv")
data.head()

data.describe()

y = data["price"]
x1 = data["size"]

plt.scatter(x1, y)
plt.xlabel("size of real estate in sqm", fontsize=20)
plt.ylabel("price of real estate in USD", fontsize=20)
plt.show()

x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
results.summary()

plt.scatter(x1, y)
yhat = 223.1787*x1+1.019e+05
fig = plt.plot(x1, yhat, lw=3.5, c="green", label="Regression Line")
plt.xlabel("size of real estate in sqm", fontsize=20)
plt.ylabel("price of real estate in USD", fontsize=20)
plt.xlim(0)
plt.ylim(0)
plt.show()
