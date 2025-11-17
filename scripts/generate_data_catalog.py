
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import hashlib
import gzip
import sys
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd

# try to import pyarrow for fast parquet metadata reading
try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:
    pq = None


def file_hash(path: Path, block_size: int = 65536) -> Optional[str]:
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def count_text_lines(path: Path) -> Optional[int]:
    """Fast count of newline characters. Works with gzip if needed."""
    try:
        if path.name.lower().endswith(".gz"):
            with gzip.open(str(path), "rb") as f:
                cnt = 0
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    cnt += chunk.count(b"\n")
            return cnt
        else:
            with path.open("rb") as f:
                cnt = 0
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    cnt += chunk.count(b"\n")
            return cnt
    except Exception:
        return None


def get_parquet_shape(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (rows, columns) for parquet using pyarrow metadata if available."""
    if pq is None:
        return None, None
    try:
        pf = pq.ParquetFile(str(path))
        return int(pf.metadata.num_rows), int(pf.metadata.num_columns)
    except Exception:
        return None, None


def sample_table(path: Path, nrows: int = 500) -> Tuple[Optional[List[str]], Optional[Dict[str, str]], Optional[List[Dict[str, Any]]]]:
    """
    Read a small sample to infer column names, dtypes and small sample_data.
    Returns: (columns, dtypes, sample_data)
    """
    try:
        name = path.name.lower()
        if name.endswith((".csv", ".csv.gz", ".tsv", ".txt")):
            df = pd.read_csv(str(path), nrows=nrows, low_memory=False, compression='infer')
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(str(path), nrows=nrows)
        elif name.endswith(".parquet"):
            # pandas may read whole file; try reading with pyarrow via pandas if available
            df = pd.read_parquet(str(path))
            if len(df) > nrows:
                df = df.head(nrows)
        elif name.endswith((".jsonl", ".ndjson", ".json")):
            df = pd.read_json(str(path), lines=True, nrows=nrows)
        else:
            return None, None, None
        cols = list(df.columns)
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        sample_data = df.head(3).to_dict(orient='records')
        return cols, dtypes, sample_data
    except Exception as e:
        return None, {"error": str(e)}, None


def get_file_metadata(path: Path) -> Dict[str, Any]:
    """Filesystem metadata"""
    try:
        st = path.stat()
        return {
            "size_bytes": int(st.st_size),
            "size_readable": f"{st.st_size / 1024:.2f} KB",
            "last_modified": datetime.fromtimestamp(st.st_mtime).isoformat()
        }
    except Exception as e:
        return {"size_bytes": None, "size_readable": None, "last_modified": None, "notes": f"stat_error:{e}"}


def get_data_description(file_name: str, relative_path: Path) -> Dict[str, str]:
    """User-provided descriptions — расширяй по необходимости"""
    descriptions = {
        "Walmart_Sales.csv": {
            "purpose": "Основной набор данных о продажах Walmart",
            "source": "Исходные данные проекта",
            "structure": "Еженедельные продажи по магазинам с метео и экономическими показателями",
            "freshness": "Обновляется при загрузке новых файлов"
        },
        "high_amount_sales": {
            "purpose": "Отфильтрованные данные о крупных продажах",
            "source": "Результат обработки Walmart_Sales.csv",
            "structure": "Подмножество с Weekly_Sales > 100",
            "freshness": "Генерируется ETL-процессом"
        }
    }
    for key, v in descriptions.items():
        if key.lower() in file_name.lower() or key.lower() in str(relative_path).lower():
            return v
    # default
    return {
        "purpose": "Автоматически сгенерированные/неописанные данные",
        "source": "ETL/проект",
        "structure": "Требует уточнения",
        "freshness": "Автообновляется"
    }


def analyze_data_file(path: Path, sample_rows: int = 500, do_full_count: bool = False) -> Dict[str, Any]:
    """Analyze data file: sample columns/types and optionally count lines."""
    name = path.name.lower()
    meta_res: Dict[str, Any] = {}
    try:
        cols, dtypes, sample_data = sample_table(path, nrows=sample_rows)
        meta_res["sample_rows_used"] = sample_rows
        if cols is not None:
            meta_res["columns_list"] = cols
            meta_res["data_types_sample"] = dtypes
            meta_res["sample_data"] = sample_data
            # If user requests full count and file is text-like, count lines
            if do_full_count and name.endswith((".csv", ".csv.gz", ".tsv", ".txt", ".jsonl", ".ndjson")):
                cnt = count_text_lines(path)
                # If file has header, cnt includes header — leave raw count; consumer decides
                meta_res["records_count"] = int(cnt) if cnt is not None else None
            else:
                # If not counting full file — set records_count to sample length (informative)
                try:
                    meta_res["records_count"] = None if sample_data is None else len(sample_data)
                except Exception:
                    meta_res["records_count"] = None
        else:
            # for non-tabular or error in sample_table, try parquet metadata
            if name.endswith(".parquet"):
                rows, cols_count = get_parquet_shape(path)
                meta_res["records_count"] = rows
                meta_res["columns_count"] = cols_count
            else:
                meta_res["records_count"] = None
    except Exception as e:
        meta_res["error"] = str(e)
    return meta_res


def generate_data_catalog(project_root: Optional[Path] = None,
                          data_dirs: Optional[List[str]] = None,
                          sample_rows: int = 500,
                          do_full_count: bool = False,
                          out_csv: Optional[Path] = None,
                          out_json: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """
    Generate a data catalog for specified directories under project_root.
    """
    if project_root is None:
        # prefer working directory (sensible for notebooks)
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if data_dirs is None:
        data_dirs = ["data", "output", "notebooks"]

    if out_csv is None:
        out_csv = project_root / "data_catalog.csv"
    if out_json is None:
        out_json = project_root / "data_catalog.json"

    print(f"Project root: {project_root}")
    catalog_records: List[Dict[str, Any]] = []

    for d in data_dirs:
        dir_path = project_root / d
        if not dir_path.exists():
            print(f"  - Directory not found, skipping: {dir_path}")
            continue
        print(f"Scanning: {dir_path}")
        for fp in dir_path.rglob("*"):
            if not fp.is_file():
                continue
            if fp.name.startswith("."):
                continue
            # skip temporary files
            lower = fp.name.lower()
            if any(tok in lower for tok in ("~$", ".tmp", ".temp", ".swp")):
                continue

            rel = fp.relative_to(project_root)
            fs_meta = get_file_metadata(fp)
            analysis = analyze_data_file(fp, sample_rows=sample_rows, do_full_count=do_full_count)
            desc = get_data_description(fp.name, rel)

            rec: Dict[str, Any] = {
                "file_name": fp.name,
                "relative_path": str(rel),
                "file_format": fp.suffix.lower(),
                "directory": d,
                "size_bytes": fs_meta.get("size_bytes"),
                "size_readable": fs_meta.get("size_readable"),
                "last_modified": fs_meta.get("last_modified"),
                "hash_md5": file_hash(fp),
                "catalog_generated": datetime.utcnow().isoformat()
            }

            # merge analysis results
            if analysis:
                # columns_list and data_types_sample may be present
                if "columns_list" in analysis and analysis["columns_list"] is not None:
                    rec["columns_list"] = analysis.get("columns_list")
                    rec["data_types_sample"] = analysis.get("data_types_sample")
                else:
                    rec["columns_list"] = []
                    rec["data_types_sample"] = {}

                # records_count may be exact (if do_full_count) or sample-based or None
                rec["records_count"] = analysis.get("records_count")
                rec["sample_rows_used"] = analysis.get("sample_rows_used")
                rec["sample_data"] = analysis.get("sample_data")
                if "error" in analysis:
                    rec["notes"] = analysis["error"]
                else:
                    rec["notes"] = ""
            else:
                rec["columns_list"] = []
                rec["data_types_sample"] = {}
                rec["records_count"] = None
                rec["sample_rows_used"] = None
                rec["sample_data"] = None
                rec["notes"] = ""

            # add user-provided description fields
            rec.update(desc)

            catalog_records.append(rec)
            print(f"  + {fp.name}")

    if not catalog_records:
        print("No files found in specified directories. Writing empty catalog.")
        # create empty frame with expected columns
        df_empty = pd.DataFrame(columns=[
            "file_name","relative_path","file_format","directory","size_bytes","size_readable","last_modified",
            "hash_md5","catalog_generated","columns_list","data_types_sample","records_count","sample_rows_used","sample_data",
            "purpose","source","structure","freshness","notes"
        ])
        df_empty.to_csv(out_csv, index=False)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return df_empty

    # normalize / prepare DataFrame
    df = pd.DataFrame(catalog_records)

    # serialize complex columns for CSV
    df["columns_list"] = df["columns_list"].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "[]")
    df["data_types_sample"] = df["data_types_sample"].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "{}")
    df["sample_data"] = df["sample_data"].apply(lambda x: json.dumps(x, ensure_ascii=False) if x else "[]")

    # save CSV and JSON
    df.to_csv(out_csv, index=False, encoding="utf-8")
    # JSON: write raw records (not serialized strings) for easier programmatic consumption
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(catalog_records, f, ensure_ascii=False, indent=2, default=str)

    print(f"Catalog written: {out_csv} ({len(df)} entries) and {out_json}")
    return df


# helper for notebooks
def generate_catalog_in_notebook(project_root: Optional[Path] = None) -> Optional[pd.DataFrame]:
    print("Generating data catalog (notebook mode)...")
    df = generate_data_catalog(project_root=project_root)
    if df is None or df.empty:
        print("No catalog generated.")
        return df
    # basic summary
    print("\nCatalog summary:")
    print(f" • Total files: {len(df)}")
    # compute formats safely
    if "file_format" in df.columns:
        formats = df["file_format"].value_counts().to_dict()
        print(f" • File formats: {formats}")
    if "size_bytes" in df.columns:
        total_mb = df["size_bytes"].dropna().sum() / 1024 / 1024
        print(f" • Total size: {total_mb:.2f} MB")
    return df


if __name__ == "__main__":
    # simple CLI for convenience if someone runs the file directly
    import argparse
    parser = argparse.ArgumentParser(description="Generate data catalog for project dirs (data, output, notebooks).")
    parser.add_argument("--project-root", default=None, help="Project root directory (default: cwd)")
    parser.add_argument("--sample-rows", type=int, default=500, help="Rows to sample for dtype inference")
    parser.add_argument("--count-rows", action="store_true", help="Do full row counts for text files (can be slow)")
    parser.add_argument("--out-csv", default=None, help="Output CSV path (default: <project_root>/data_catalog.csv)")
    parser.add_argument("--out-json", default=None, help="Output JSON path (default: <project_root>/data_catalog.json)")
    args, unknown = parser.parse_known_args()
    try:
        pr = Path(args.project_root) if args.project_root else None
        out_csv = Path(args.out_csv) if args.out_csv else None
        out_json = Path(args.out_json) if args.out_json else None
        df = generate_data_catalog(project_root=pr,
                                   sample_rows=args.sample_rows,
                                   do_full_count=args.count_rows,
                                   out_csv=out_csv,
                                   out_json=out_json)
        if df is not None:
            print(f"Finished: {len(df)} entries")
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
