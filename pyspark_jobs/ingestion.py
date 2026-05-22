from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("SupplyChainIngestion") \
    .getOrCreate()

# Read dataset
df = spark.read.csv(
    "data/DataCoSupplyChainDataset.csv",
    header=True,
    inferSchema=True
)

print("Rows:", df.count())
print("Columns:", len(df.columns))

df.show(5, vertical=True, truncate=False)