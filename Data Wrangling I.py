import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Shape of dataset
print("Shape:", df.shape)

# Column names
print(df.columns)

# Data types
print(df.dtypes)

# Missing values
print(df.isnull().sum())

# Statistical information
print(df.describe())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Type conversion
df['Age'] = df['Age'].astype(int)

# Normalization
df['Fare_Normalized'] = (
    (df['Fare'] - df['Fare'].min()) /
    (df['Fare'].max() - df['Fare'].min())
)

# Convert categorical to numerical
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# Final dataset
print(df.head())
