import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv")

# Fix 1: drop rows with missing or invalid values before fitting
df_clean = df[
    df["units_sold"].notna() &
    (df["ad_spend"] > 0) &
    (df["price"] > 0)
].copy()

X = df_clean[["month", "ad_spend", "price", "competitor_price"]]
y = df_clean["units_sold"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(y, model.predict(X))
plt.xlabel("Actual units_sold")
plt.ylabel("Predicted units_sold")
plt.title("Actual vs Predicted")

# Fix 2: plt.savefig(), not plt.save()
plt.savefig("regression_plot.png")
