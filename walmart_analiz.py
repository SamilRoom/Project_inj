from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Example - Sales Data") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

df = spark.read.csv(
    "Walmart_Sales.csv",
    header=True,
    inferSchema=True,
    sep=","
)

df.show(10, truncate=False)
df.printSchema()

df.select("Store", "Date", "Weekly_Sales").show(10)

df = df.withColumn("Weekly_Sales", col("Weekly_Sales").cast("double"))

high_amount = df.filter(col("Weekly_Sales") > 100)
high_amount.show(10)

df.orderBy(col("Weekly_Sales").desc()).show(10)

df.describe().show()

(high_amount
 .repartition(1)
 .write
 .option("header", True)
 .mode("overwrite")
 .csv("output/high_amount_sales")
)

spark.stop()
