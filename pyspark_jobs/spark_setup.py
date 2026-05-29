import os
import shutil
import sys
from pathlib import Path


def configure_local_pyspark() -> None:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def configure_spark_builder(builder):
    return (
        builder.master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "2")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
        .config("spark.sql.parquet.output.committer.class", "org.apache.parquet.hadoop.ParquetOutputCommitter")
    )


def write_json_lines(df, output_path: Path) -> int:
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    row_count = 0
    part_number = 0
    rows_per_file = 50000
    handle = None

    try:
        for line in df.toJSON().toLocalIterator():
            if row_count % rows_per_file == 0:
                if handle is not None:
                    handle.close()
                part_path = output_path / f"part-{part_number:05d}.json"
                handle = part_path.open("w", encoding="utf-8")
                part_number += 1

            handle.write(line)
            handle.write("\n")
            row_count += 1
    finally:
        if handle is not None:
            handle.close()

    return row_count


def json_part_files(input_path: Path) -> list[str]:
    files = sorted(Path(input_path).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON part files found in {input_path}")
    return [str(path) for path in files]
