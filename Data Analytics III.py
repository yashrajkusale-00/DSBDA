import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Load dataset
df = pd.read_csv("iris.csv")

# Display dataset
print(df.head())

# Features and target
X = df.iloc[:, 0:4]
y = df['species']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print(y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Error rate
error_rate = 1 - accuracy
print("Error Rate:", error_rate)

# Precision
precision = precision_score(
    y_test,
    y_pred,
    average='macro'
)

print("Precision:", precision)

# Recall
recall = recall_score(
    y_test,
    y_pred,
    average='macro'
)

print("Recall:", recall)
