import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/monthly_sales.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())
X = df["previous_month_sales"].values
y = df["sales_revenue"].values

print("\nFirst 5 X values:")
print(X[:5])

print("\nFirst 5 y values:")
print(y[:5])


plt.scatter(X, y)

plt.xlabel("Previous Month Sales")
plt.ylabel("Next Month Sales")
plt.title("ERP Sales Relationship")

plt.show()