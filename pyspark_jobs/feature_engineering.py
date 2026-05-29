from pathlib import Path

from spark_setup import (
    configure_local_pyspark,
    configure_spark_builder,
    json_part_files,
    write_json_lines,
)

configure_local_pyspark()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, quarter, substring, to_timestamp, when, year


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "processed_data" / "orders_clean"
OUTPUT_PATH = ROOT_DIR / "processed_data" / "features"


def build_spark_session() -> SparkSession:
    return configure_spark_builder(
        SparkSession.builder.appName("SupplyChainFeatureEngineering")
    ).getOrCreate()


def add_features(df):
    return (
        df.withColumn(
            "order_date",
            to_timestamp(substring(col("order_date"), 1, 19), "yyyy-MM-dd'T'HH:mm:ss"),
        )
        .withColumn("order_year", year(col("order_date")))
        .withColumn("order_month", month(col("order_date")))
        .withColumn("order_quarter", quarter(col("order_date")))
        .withColumn(
            "delivery_performance",
            when(col("shipping_delay") <= 0, "On or Ahead").otherwise("Delayed"),
        )
        .withColumn(
            "order_value_segment",
            when(col("order_item_total") >= 500, "High Value")
            .when(col("order_item_total") >= 100, "Medium Value")
            .otherwise("Low Value"),
        )
        .withColumn(
            "profit_margin",
            when(col("sales_per_customer") != 0, col("benefit_per_order") / col("sales_per_customer"))
            .otherwise(0.0),
        )
        .withColumn(
            "shipping_efficiency",
            when(col("scheduled_shipping_days") != 0, col("real_shipping_days") / col("scheduled_shipping_days"))
            .otherwise(None),
        )
        .withColumn("is_loss_order", when(col("benefit_per_order") < 0, 1).otherwise(0))
    )


def main() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.json(json_part_files(INPUT_PATH))
    feature_df = add_features(df).repartition(8).cache()

    feature_df.show(5, vertical=True, truncate=False)
    written_rows = write_json_lines(feature_df, OUTPUT_PATH)

    print("Feature Row Count:", feature_df.count())
    print(f"Rows Written: {written_rows}")
    print(f"Features Saved Successfully: {OUTPUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
