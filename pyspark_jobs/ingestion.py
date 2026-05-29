from pathlib import Path

from spark_setup import configure_local_pyspark, configure_spark_builder

configure_local_pyspark()

from pyspark.sql import SparkSession


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "DataCoSupplyChainDataset.csv"


def build_spark_session() -> SparkSession:
    return configure_spark_builder(
        SparkSession.builder.appName("SupplyChainIngestion")
    ).getOrCreate()


def load_raw_orders(spark: SparkSession):
    return spark.read.csv(
        str(DATA_PATH),
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
    )


def main() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_raw_orders(spark)

    print("Rows:", df.count())
    print("Columns:", len(df.columns))
    df.printSchema()
    df.show(5, vertical=True, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
