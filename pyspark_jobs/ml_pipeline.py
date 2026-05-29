from pathlib import Path

from spark_setup import (
    configure_local_pyspark,
    configure_spark_builder,
    json_part_files,
    write_json_lines,
)

configure_local_pyspark()

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "processed_data" / "features"
MODEL_PATH = ROOT_DIR / "models" / "late_delivery_risk_pipeline"
PREDICTION_PATH = ROOT_DIR / "processed_data" / "predictions"

CATEGORICAL_COLUMNS = [
    "category_name",
    "customer_segment",
    "order_country",
    "shipping_mode",
    "payment_type",
]

NUMERIC_COLUMNS = [
    "order_item_quantity",
    "order_item_total",
    "real_shipping_days",
    "scheduled_shipping_days",
    "benefit_per_order",
    "sales_per_customer",
    "shipping_delay",
    "profit_margin",
    "shipping_efficiency",
    "order_month",
    "order_quarter",
]


def build_spark_session() -> SparkSession:
    return configure_spark_builder(
        SparkSession.builder.appName("SupplyChainMLPipeline")
    ).getOrCreate()


def build_pipeline() -> Pipeline:
    indexers = [
        StringIndexer(
            inputCol=column_name,
            outputCol=f"{column_name}_idx",
            handleInvalid="keep",
        )
        for column_name in CATEGORICAL_COLUMNS
    ]

    encoders = [
        OneHotEncoder(
            inputCol=f"{column_name}_idx",
            outputCol=f"{column_name}_vec",
            handleInvalid="keep",
        )
        for column_name in CATEGORICAL_COLUMNS
    ]

    feature_columns = NUMERIC_COLUMNS + [
        f"{column_name}_vec" for column_name in CATEGORICAL_COLUMNS
    ]

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features",
        handleInvalid="keep",
    )

    classifier = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=40,
        maxDepth=8,
        seed=42,
    )

    return Pipeline(stages=indexers + encoders + [assembler, classifier])


def main() -> None:
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = (
        spark.read.json(json_part_files(INPUT_PATH))
        .withColumn("label", col("late_delivery_risk").cast("double"))
        .dropna(subset=["label", *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS])
    )

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = build_pipeline().fit(train_df)
    predictions = model.transform(test_df).cache()

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)
    area_under_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Area Under ROC: {area_under_roc:.4f}")

    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    metrics_df = spark.createDataFrame(
        [(float(accuracy), float(area_under_roc))],
        ["accuracy", "area_under_roc"],
    )
    write_json_lines(metrics_df.coalesce(1), MODEL_PATH)
    write_json_lines(
        predictions.select(
            "category_name",
            "order_country",
            "shipping_mode",
            "late_delivery_risk",
            col("prediction").cast("int").alias("prediction"),
        ),
        PREDICTION_PATH,
    )

    print(f"Model Metrics Saved Successfully: {MODEL_PATH}")
    print(f"Predictions Saved Successfully: {PREDICTION_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
