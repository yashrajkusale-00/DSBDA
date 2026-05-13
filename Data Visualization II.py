import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display dataset
print(df.head())

# Dataset information
print(df.info())

# Boxplot
plt.figure(figsize=(10,6))

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=df
)

plt.title("Age Distribution by Gender and Survival")

plt.xlabel("Gender")

plt.ylabel("Age")

plt.show()

### SCALA Spark Program

import org.apache.spark.sql.SparkSession

object TitanicSparkApp {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("Titanic Spark Program")
            .master("local[*]")
            .getOrCreate()

        import spark.implicits._

        // Sample Titanic data
        val data = Seq(
            ("Male", 22, 0),
            ("Female", 38, 1),
            ("Female", 26, 1),
            ("Male", 35, 0)
        )

        // Create DataFrame
        val df = data.toDF("Sex", "Age", "Survived")

        // Display DataFrame
        df.show()

        spark.stop()
    }
}
