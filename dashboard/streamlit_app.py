import os
import sys
from pathlib import Path

import streamlit as st

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, count, desc, max, min, round, sum


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT_DIR / "processed_data" / "features"

st.set_page_config(
    page_title="Smart Supply Chain Dashboard",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e8ef;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569;
    }
    .section-title {
        margin-top: 0.25rem;
        margin-bottom: 0.4rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
    }
    .subtle {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border: 1px solid #bbf7d0;
        background: #f0fdf4;
        color: #166534;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def json_part_files(input_path: Path) -> list[str]:
    files = sorted(input_path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON part files found in {input_path}")
    return [str(path) for path in files]


@st.cache_resource
def get_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder.appName("SupplyChainStreamlitDashboard")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_data(show_spinner=False)
def load_filter_options() -> dict:
    spark = get_spark_session()
    df = spark.read.json(json_part_files(FEATURE_PATH))

    countries = [
        row["order_country"]
        for row in df.select("order_country")
        .dropna()
        .distinct()
        .orderBy("order_country")
        .collect()
    ]
    categories = [
        row["category_name"]
        for row in df.select("category_name")
        .dropna()
        .distinct()
        .orderBy("category_name")
        .collect()
    ]
    years = [
        int(row["order_year"])
        for row in df.select("order_year").dropna().distinct().orderBy("order_year").collect()
    ]

    return {
        "countries": countries,
        "categories": categories,
        "years": years,
    }


def apply_filters(df, countries: list[str], categories: list[str], years: list[int]):
    if countries:
        df = df.filter(col("order_country").isin(countries))
    if categories:
        df = df.filter(col("category_name").isin(categories))
    if years:
        df = df.filter(col("order_year").isin(years))
    return df


@st.cache_data(show_spinner="Loading dashboard data...")
def load_dashboard_data(
    selected_countries: tuple[str, ...],
    selected_categories: tuple[str, ...],
    selected_years: tuple[int, ...],
) -> dict:
    spark = get_spark_session()
    df = spark.read.json(json_part_files(FEATURE_PATH))
    df = apply_filters(df, list(selected_countries), list(selected_categories), list(selected_years)).cache()

    totals = df.agg(
        count("*").alias("total_orders"),
        round(sum("sales_per_customer"), 2).alias("total_sales"),
        round(avg("late_delivery_risk"), 4).alias("late_delivery_rate"),
        round(avg("shipping_delay"), 2).alias("avg_shipping_delay"),
        round(avg("profit_margin"), 4).alias("avg_profit_margin"),
    ).collect()[0]

    country_count = df.select("order_country").dropna().distinct().count()

    category_sales = [
        row.asDict()
        for row in df.groupBy("category_name")
        .agg(round(sum("sales_per_customer"), 2).alias("sales"))
        .orderBy(desc("sales"))
        .limit(10)
        .collect()
    ]

    delivery_status = [
        row.asDict()
        for row in df.groupBy("delivery_status")
        .agg(count("*").alias("orders"))
        .orderBy(desc("orders"))
        .collect()
    ]

    country_performance = [
        row.asDict()
        for row in df.groupBy("order_country")
        .agg(
            count("*").alias("orders"),
            round(sum("sales_per_customer"), 2).alias("sales"),
            round(avg("late_delivery_risk"), 4).alias("late_delivery_rate"),
        )
        .orderBy(desc("orders"))
        .limit(12)
        .collect()
    ]

    monthly_sales = [
        row.asDict()
        for row in df.groupBy("order_year", "order_month")
        .agg(
            count("*").alias("orders"),
            round(sum("sales_per_customer"), 2).alias("sales"),
        )
        .orderBy("order_year", "order_month")
        .collect()
    ]

    risk_by_mode = [
        row.asDict()
        for row in df.groupBy("shipping_mode")
        .agg(
            count("*").alias("orders"),
            round(avg("late_delivery_risk"), 4).alias("late_delivery_rate"),
            round(avg("shipping_delay"), 2).alias("avg_delay"),
        )
        .orderBy(desc("late_delivery_rate"))
        .collect()
    ]

    sample_orders = [
        row.asDict()
        for row in df.select(
            "order_date",
            "category_name",
            "order_country",
            "shipping_mode",
            "delivery_status",
            "sales_per_customer",
            "shipping_delay",
            "profit_margin",
        )
        .orderBy(desc("sales_per_customer"))
        .limit(25)
        .collect()
    ]

    return {
        "total_orders": int(totals["total_orders"] or 0),
        "total_sales": float(totals["total_sales"] or 0),
        "late_delivery_rate": float(totals["late_delivery_rate"] or 0),
        "avg_shipping_delay": float(totals["avg_shipping_delay"] or 0),
        "avg_profit_margin": float(totals["avg_profit_margin"] or 0),
        "country_count": int(country_count),
        "category_sales": category_sales,
        "delivery_status": delivery_status,
        "country_performance": country_performance,
        "monthly_sales": monthly_sales,
        "risk_by_mode": risk_by_mode,
        "sample_orders": sample_orders,
    }


if not FEATURE_PATH.exists() or not list(FEATURE_PATH.glob("*.json")):
    st.error("Feature data is missing. Run preprocessing.py and feature_engineering.py first.")
    st.stop()

filter_options = load_filter_options()

with st.sidebar:
    st.header("Controls")
    selected_years = st.multiselect(
        "Order year",
        filter_options["years"],
        default=filter_options["years"],
    )
    selected_countries = st.multiselect(
        "Order country",
        filter_options["countries"],
        default=[],
        placeholder="All countries",
    )
    selected_categories = st.multiselect(
        "Category",
        filter_options["categories"],
        default=[],
        placeholder="All categories",
    )
    st.divider()
    st.caption(f"Source: {FEATURE_PATH.name}")
    st.caption("Engine: Python + PySpark")

dashboard_data = load_dashboard_data(
    tuple(selected_countries),
    tuple(selected_categories),
    tuple(selected_years),
)

left, right = st.columns([0.72, 0.28], vertical_alignment="center")
with left:
    st.title("Smart Supply Chain Dashboard")
    st.markdown(
        '<div class="subtle">Monitor orders, sales, delivery risk, and shipment performance from the processed supply chain data.</div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown('<span class="status-pill">Processed data ready</span>', unsafe_allow_html=True)

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Orders", f"{dashboard_data['total_orders']:,}")
metric_2.metric("Sales", f"${dashboard_data['total_sales']:,.0f}")
metric_3.metric("Countries", f"{dashboard_data['country_count']:,}")
metric_4.metric("Late Risk", f"{dashboard_data['late_delivery_rate']:.1%}")
metric_5.metric("Avg Delay", f"{dashboard_data['avg_shipping_delay']:.2f} days")

st.divider()

chart_a, chart_b = st.columns([0.56, 0.44])
with chart_a:
    st.markdown('<div class="section-title">Top Categories by Sales</div>', unsafe_allow_html=True)
    st.bar_chart(
        dashboard_data["category_sales"],
        x="category_name",
        y="sales",
        color="#2563eb",
        use_container_width=True,
    )

with chart_b:
    st.markdown('<div class="section-title">Delivery Status Mix</div>', unsafe_allow_html=True)
    st.bar_chart(
        dashboard_data["delivery_status"],
        x="delivery_status",
        y="orders",
        color="#059669",
        use_container_width=True,
    )

chart_c, chart_d = st.columns([0.52, 0.48])
with chart_c:
    st.markdown('<div class="section-title">Monthly Sales Trend</div>', unsafe_allow_html=True)
    st.line_chart(
        dashboard_data["monthly_sales"],
        x="order_month",
        y="sales",
        color="#7c3aed",
        use_container_width=True,
    )

with chart_d:
    st.markdown('<div class="section-title">Late Delivery Risk by Shipping Mode</div>', unsafe_allow_html=True)
    st.bar_chart(
        dashboard_data["risk_by_mode"],
        x="shipping_mode",
        y="late_delivery_rate",
        color="#dc2626",
        use_container_width=True,
    )

st.markdown('<div class="section-title">Country Performance</div>', unsafe_allow_html=True)
st.dataframe(
    dashboard_data["country_performance"],
    use_container_width=True,
    hide_index=True,
)

st.markdown('<div class="section-title">High Value Order Preview</div>', unsafe_allow_html=True)
st.dataframe(
    dashboard_data["sample_orders"],
    use_container_width=True,
    hide_index=True,
)
