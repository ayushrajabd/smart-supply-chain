from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count

# -------------------------------
# Spark Session
# -------------------------------

spark = SparkSession.builder \
    .appName("SupplyChainPreprocessing") \
    .getOrCreate()

# -------------------------------
# Read Dataset
# -------------------------------

df = spark.read.csv(
    "data/DataCoSupplyChainDataset.csv",
    header=True,
    inferSchema=True
)

# -------------------------------
# Select Important Columns
# -------------------------------

required_cols = [
    "Category Name",
    "Customer City",
    "Customer Country",
    "Sales per customer",
    "Order Item Quantity",
    "Order Status",
    "Shipping Mode",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)"
]

df = df.select(required_cols)

# -------------------------------
# Cache Dataset
# -------------------------------

df.cache()

# -------------------------------
# Null Value Analysis
# -------------------------------

print("NULL VALUES")

df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show()

# -------------------------------
# Remove Duplicates
# -------------------------------

before = df.count()

df = df.dropDuplicates()

after = df.count()

print("Duplicates Removed:", before - after)

# -------------------------------
# Remove Null Rows
# -------------------------------

df = df.dropna()

print("Final Rows:", df.count())

# -------------------------------
# Save Optimized Data
# -------------------------------

df.write \
    .mode("overwrite") \
    .partitionBy("Customer Country") \
    .parquet("processed_data")

print("Preprocessing Completed")