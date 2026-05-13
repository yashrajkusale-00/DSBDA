import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('iris')

# Display dataset
print(df.head())

# Dataset information
print(df.info())

# Data types
print(df.dtypes)

# Histograms
df.hist(figsize=(10,8))

plt.show()

# Histogram for sepal length
sns.histplot(df['sepal_length'], bins=20)

plt.title("Sepal Length Distribution")

plt.show()

# Histogram for petal length
sns.histplot(df['petal_length'], bins=20)

plt.title("Petal Length Distribution")

plt.show()

# Boxplot for all features
plt.figure(figsize=(10,6))

sns.boxplot(data=df)

plt.title("Boxplot of Iris Features")

plt.show()

# Individual boxplots
sns.boxplot(y=df['sepal_width'])

plt.title("Boxplot of Sepal Width")

plt.show()

sns.boxplot(y=df['petal_width'])

plt.title("Boxplot of Petal Width")

plt.show()

# Pairplot
sns.pairplot(df, hue='species')

plt.show()


### SCALA Spark Program

import org.apache.spark.sql.SparkSession

object IrisSparkApp {

    def main(args: Array[String]): Unit = {

        val spark = SparkSession
            .builder()
            .appName("Iris Spark Program")
            .master("local[*]")
            .getOrCreate()

        import spark.implicits._

        // Sample Iris data
        val data = Seq(
            (5.1, 3.5, 1.4, 0.2, "setosa"),
            (7.0, 3.2, 4.7, 1.4, "versicolor"),
            (6.3, 3.3, 6.0, 2.5, "virginica")
        )

        // Create DataFrame
        val df = data.toDF(
            "SepalLength",
            "SepalWidth",
            "PetalLength",
            "PetalWidth",
            "Species"
        )

        // Display DataFrame
        df.show()

        spark.stop()
    }
}
