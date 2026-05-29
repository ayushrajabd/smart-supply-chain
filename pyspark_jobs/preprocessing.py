from pathlib import Path

from spark_setup import configure_local_pyspark, configure_spark_builder, write_json_lines

configure_local_pyspark()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, to_timestamp, trim, when
from pyspark.sql.types import DoubleType, IntegerType


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "DataCoSupplyChainDataset.csv"
OUTPUT_PATH = ROOT_DIR / "processed_data" / "orders_clean"

REQUIRED_COLUMNS = [
    "Category Name",
    "Customer City",
    "Customer Country",
    "Customer Segment",
    "Order City",
    "Order Country",
    "Order Item Quantity",
    "Order Item Total",
    "Order Status",
    "Shipping Mode",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Type",
    "order date (DateOrders)",
]

RENAME_COLUMNS = {
    "Category Name": "category_name",
    "Customer City": "customer_city",
    "Customer Country": "customer_country",
    "Customer Segment": "customer_segment",
    "Order City": "order_city",
    "Order Country": "order_country",
    "Order Item Quantity": "order_item_quantity",
    "Order Item Total": "order_item_total",
    "Order Status": "order_status",
    "Shipping Mode": "shipping_mode",
    "Delivery Status": "delivery_status",
    "Late_delivery_risk": "late_delivery_risk",
    "Days for shipping (real)": "real_shipping_days",
    "Days for shipment (scheduled)": "scheduled_shipping_days",
    "Benefit per order": "benefit_per_order",
    "Sales per customer": "sales_per_customer",
    "Type": "payment_type",
    "order date (DateOrders)": "order_date",
}

NUMERIC_COLUMNS = {
    "order_item_quantity": IntegerType(),
    "order_item_total": DoubleType(),
    "late_delivery_risk": IntegerType(),
    "real_shipping_days": IntegerType(),
    "scheduled_shipping_days": IntegerType(),
    "benefit_per_order": DoubleType(),
    "sales_per_customer": DoubleType(),
}

TEXT_COLUMNS = [
    "category_name",
    "customer_city",
    "customer_country",
    "customer_segment",
    "order_city",
    "order_country",
    "order_status",
    "shipping_mode",
    "delivery_status",
    "payment_type",
]


def build_spark_session() -> SparkSession:
    return configure_spark_builder(
        SparkSession.builder.appName("SupplyChainPreprocessing")
    ).getOrCreate()


def load_raw_orders(spark: SparkSession):
    return spark.read.csv(
        str(DATA_PATH),
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
    )


def preprocess_orders(df):
    df = df.select(*REQUIRED_COLUMNS)

    for old_name, new_name in RENAME_COLUMNS.items():
        df = df.withColumnRenamed(old_name, new_name)

    for column_name in TEXT_COLUMNS:
        df = df.withColumn(column_name, trim(col(column_name)))

    for column_name, spark_type in NUMERIC_COLUMNS.items():
        df = df.withColumn(column_name, col(column_name).cast(spark_type))

    df = df.withColumn("order_date", to_timestamp(col("order_date"), "M/d/yyyy H:mm"))

    before_count = df.count()
    df = df.dropDuplicates()
    after_count = df.count()
    print(f"Duplicates Removed: {before_count - after_count}")

    print("Null Values Before Critical Drop")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show(
        truncate=False
    )

    critical_columns = [
        "category_name",
        "customer_country",
        "sales_per_customer",
        "delivery_status",
        "late_delivery_risk",
        "order_date",
    ]
    df = df.dropna(subset=critical_columns)

    df = df.withColumn(
        "shipping_delay", col("real_shipping_days") - col("scheduled_shipping_days")
    )
    df = df.withColumn(
        "high_risk_delivery",
        when(col("late_delivery_risk") == 1, "Yes").otherwise("No"),
    )
    df = df.withColumn(
        "profit_category",
        when(col("benefit_per_order") > 100, "High Profit")
        .when(col("benefit_per_order") > 0, "Medium Profit")
        .otherwise("Low Profit"),
    )

    return df.repartition(8)


def main() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    print("Spark Session Started")
    df = load_raw_orders(spark)
    print("Dataset Loaded Successfully")

    clean_df = preprocess_orders(df).cache()
    clean_df.show(5, vertical=True, truncate=False)

    print("Starting Clean Data Save...")
    written_rows = write_json_lines(clean_df, OUTPUT_PATH)

    print("Final Row Count:", clean_df.count())
    print("Final Column Count:", len(clean_df.columns))
    print(f"Rows Written: {written_rows}")
    print(f"Clean Data Saved Successfully: {OUTPUT_PATH}")
    print("Preprocessing Pipeline Completed Successfully")

    spark.stop()


if __name__ == "__main__":
    main()
