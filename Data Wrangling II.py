import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create dataset
data = {
    'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male',
               'Female', 'Male', 'Female', 'Male', 'Female'],
    'Maths': [85, 90, np.nan, 35, 95, 88, 76, 500, 65, 80],
    'Science': [78, 85, 89, 45, np.nan, 92, 70, 88, 60, 75],
    'English': [80, 95, 78, 40, 85, 91, 72, 85, 68, np.nan]
}

df = pd.DataFrame(data)

# Missing values
print(df.isnull().sum())

# Fill missing values
df['Maths'] = df['Maths'].fillna(df['Maths'].mean())
df['Science'] = df['Science'].fillna(df['Science'].mean())
df['English'] = df['English'].fillna(df['English'].mean())

# Statistics
print(df.describe())

# Boxplot
plt.boxplot(df['Maths'])
plt.title("Boxplot of Maths")
plt.show()

# Z-score
z = np.abs(stats.zscore(df['Maths']))
print(z)

# Detect outlier
print(df[z > 3])

# Replace outlier
median = df['Maths'].median()
df.loc[df['Maths'] > 100, 'Maths'] = median

# Log transformation
df['Maths_log'] = np.log(df['Maths'] + 1)

# Final data
print(df)

#Use Titanic Dataset again.

import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")

grouped = df.groupby('Sex')['Age']

print("Mean")
print(grouped.mean())

print("Median")
print(grouped.median())

print("Minimum")
print(grouped.min())

print("Maximum")
print(grouped.max())

print("Standard Deviation")
print(grouped.std())

male_age = df[df['Sex'] == 'male']['Age'].tolist()
female_age = df[df['Sex'] == 'female']['Age'].tolist()

print(male_age)
print(female_age)

#Iris Dataset Statistics

import pandas as pd

# Load dataset
df = pd.read_csv("iris.csv")

# Display first rows
print(df.head())

# Group by species
grouped = df.groupby('species')

# Mean
print(grouped.mean())

# Standard deviation
print(grouped.std())

# Percentiles
print(grouped.quantile([0.25, 0.5, 0.75]))

# Setosa
setosa = df[df['species'] == 'setosa']
print(setosa.describe())

# Versicolor
versicolor = df[df['species'] == 'versicolor']
print(versicolor.describe())

# Virginica
virginica = df[df['species'] == 'virginica']
print(virginica.describe())
