import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv")

X = df[["month", "ad_spend", "price", "competitor_price"]]
y = df["units_sold"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.scatter(y, model.predict(X))
plt.xlabel("Actual units_sold")
plt.ylabel("Predicted units_sold")
plt.title("Actual vs Predicted")
plt.save("regression_plot.png")
