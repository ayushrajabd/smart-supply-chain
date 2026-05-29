from pathlib import Path

from spark_setup import (
    configure_local_pyspark,
    configure_spark_builder,
    json_part_files,
    write_json_lines,
)

configure_local_pyspark()

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, round, sum


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "processed_data" / "features"
OUTPUT_DIR = ROOT_DIR / "processed_data" / "analytics"


def build_spark_session() -> SparkSession:
    return configure_spark_builder(
        SparkSession.builder.appName("SupplyChainAnalytics")
    ).getOrCreate()


def write_report(df, output_name: str, report_df) -> None:
    output_path = OUTPUT_DIR / output_name
    write_json_lines(report_df.coalesce(1), output_path)
    print(f"\n{output_name}")
    report_df.show(20, truncate=False)


def main() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.json(json_part_files(INPUT_PATH)).cache()

    country_report = (
        df.groupBy("order_country")
        .agg(
            count("*").alias("total_orders"),
            round(sum("order_item_total"), 2).alias("total_order_value"),
            round(avg("shipping_delay"), 2).alias("avg_shipping_delay"),
            round(avg("late_delivery_risk"), 3).alias("late_delivery_rate"),
        )
        .orderBy(desc("total_orders"))
    )

    category_report = (
        df.groupBy("category_name")
        .agg(
            count("*").alias("total_orders"),
            round(sum("sales_per_customer"), 2).alias("total_sales"),
            round(avg("profit_margin"), 4).alias("avg_profit_margin"),
        )
        .orderBy(desc("total_sales"))
    )

    delivery_report = (
        df.groupBy("delivery_status", "shipping_mode")
        .agg(
            count("*").alias("total_orders"),
            round(avg("shipping_delay"), 2).alias("avg_shipping_delay"),
            round(avg("late_delivery_risk"), 3).alias("late_delivery_rate"),
        )
        .orderBy(desc("total_orders"))
    )

    monthly_report = (
        df.groupBy("order_year", "order_month")
        .agg(
            count("*").alias("total_orders"),
            round(sum("sales_per_customer"), 2).alias("total_sales"),
            round(avg(col("is_loss_order")), 3).alias("loss_order_rate"),
        )
        .orderBy("order_year", "order_month")
    )

    write_report(df, "orders_by_country", country_report)
    write_report(df, "sales_by_category", category_report)
    write_report(df, "delivery_by_mode", delivery_report)
    write_report(df, "monthly_sales", monthly_report)

    print("Analytics Pipeline Completed Successfully")
    spark.stop()


if __name__ == "__main__":
    main()
