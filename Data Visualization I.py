import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('titanic')

# Display dataset
print(df.head())

# Dataset information
print(df.shape)
print(df.columns)
print(df.info())

# Survival count
sns.countplot(x='survived', data=df)

plt.title("Survival Count")

plt.show()

# Survival based on gender
sns.countplot(x='survived', hue='sex', data=df)

plt.title("Survival based on Gender")

plt.show()

# Survival based on class
sns.countplot(x='survived', hue='class', data=df)

plt.title("Survival based on Passenger Class")

plt.show()

# Age distribution
sns.histplot(df['age'].dropna(), bins=30)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.show()

# Fare distribution
plt.figure(figsize=(8,5))

plt.hist(df['fare'], bins=30)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Number of Passengers")

plt.show()

#### SCALA Spark Program
import org.apache.spark.sql.SparkSession

object SimpleSparkApp {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("Simple Spark Program")
            .master("local[*]")
            .getOrCreate()

        // Create sample data
        val data = Seq(
            ("Yash", 21),
            ("Rahul", 22),
            ("Amit", 20)
        )

        // Convert to DataFrame
        import spark.implicits._

        val df = data.toDF("Name", "Age")

        // Display DataFrame
        df.show()

        spark.stop()
    }
}
