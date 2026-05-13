import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("HousingData.csv")

# Display dataset
print(df.head())

# Shape
print(df.shape)

# Missing values
print(df.isnull().sum())

# Fill missing values
df = df.fillna(df.mean())

# Features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Compare actual and predicted
result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print(result.head())

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Plot graph
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()
