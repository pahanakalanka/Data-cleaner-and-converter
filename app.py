"""
app.py
======

Streamlit frontend UI router for the Data Cleaner & Converter Suite.

This file contains ONLY UI/routing logic. All core parsing/cleaning
logic lives in the backend modules:
    - pdf_cleaner.py    (process_pdf, extract_tables_digital)
    - csv_cleaner.py    (process_csv)
    - excel_cleaner.py  (process_excel)
    - converter.py      (convert_*, dataframe_to_csv_bytes, dataframe_to_excel_bytes)

New in this version:
    - Summary statistics cards (raw vs cleaned rows/cols, duplicates,
      nulls) shown after every process run.
    - A "🔍 Live Filter Options" sidebar section (NaN-column filter +
      free-text search) that dynamically updates the preview table and
      download buttons.
    - A Single File / Batch Mode toggle on every cleaner tab. Batch mode
      processes multiple uploads at once, shows a per-file progress
      table, and bundles everything into a single ZIP download via the
      local `dataframes_to_zip_bytes()` helper.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from pdf_cleaner import process_pdf, extract_tables_digital
from csv_cleaner import process_csv
from excel_cleaner import process_excel
from converter import (
    convert_pdf_to_csv,
    convert_pdf_to_excel,
    convert_excel_to_csv,
    convert_csv_to_excel,
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
)


# --------------------------------------------------------------------------- #
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Data Cleaner & Converter Suite",
    page_icon="🛠️",
    layout="wide",
)

DELIMITER_OPTIONS = {
    "Auto": None,
    "Comma ( , )": ",",
    "Semicolon ( ; )": ";",
    "Tab ( \\t )": "\t",
    "Pipe ( | )": "|",
}


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #

def _is_nonempty_df(df) -> bool:
    """Return True only if df is a real, non-empty pandas DataFrame."""
    return isinstance(df, pd.DataFrame) and not df.empty


def _finalize(df: Optional[pd.DataFrame], drop_duplicates: bool):
    """
    Take a cleaned-but-not-yet-deduplicated DataFrame and:
      - count how many duplicate rows it contains, and
      - optionally drop them, based on the caller's toggle.

    Returns (final_df, duplicate_row_count). Applied uniformly across
    PDF/CSV/Excel so the "Duplicate Rows Removed" metric means the same
    thing regardless of which cleaner produced the data.
    """
    if not _is_nonempty_df(df):
        return df, 0
    dup_count = int(df.duplicated().sum())
    final_df = df.drop_duplicates().reset_index(drop=True) if drop_duplicates else df
    return final_df, dup_count


# --------------------------------------------------------------------------- #
# "Raw" (pre-cleaning) readers — used ONLY to compute comparison stats for
# the summary dashboard. These are intentionally lightweight and never used
# as the actual cleaned output.
# --------------------------------------------------------------------------- #

def _raw_read_pdf(uploaded_file) -> Optional[pd.DataFrame]:
    """Best-effort raw table extraction (no cleaning) for stats comparison."""
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        tables = extract_tables_digital(raw_bytes)
        if not tables:
            return None
        return pd.concat(tables, ignore_index=True, sort=False)
    except Exception:
        return None


def _raw_read_csv(uploaded_file, delimiter: Optional[str]) -> Optional[pd.DataFrame]:
    """Best-effort raw CSV parse (no cleaning) for stats comparison."""
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("latin-1", errors="ignore")
        sep = delimiter if delimiter else None
        return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")
    except Exception:
        return None


def _raw_read_excel(uploaded_file, sheet_name) -> Optional[pd.DataFrame]:
    """Best-effort raw Excel sheet read (no cleaning) for stats comparison."""
    try:
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        filename = (getattr(uploaded_file, "name", "") or "").lower()
        engine = "xlrd" if filename.endswith(".xls") else "openpyxl"
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet_name, engine=engine)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Summary statistics dashboard
# --------------------------------------------------------------------------- #

def render_summary_metrics(raw_df: Optional[pd.DataFrame], dup_count: int, final_df: pd.DataFrame) -> None:
    """
    Render the 4-metric summary dashboard comparing raw vs cleaned output.
    Falls back to cleaned-only figures when a raw baseline isn't available
    (e.g. OCR-only PDFs, where no un-cleaned table exists to compare against).
    """
    if not _is_nonempty_df(final_df):
        return

    final_rows, final_cols = final_df.shape
    nulls_after = int(final_df.isna().sum().sum())

    have_raw = _is_nonempty_df(raw_df)
    if have_raw:
        raw_rows, raw_cols = raw_df.shape
        nulls_before = int(raw_df.isna().sum().sum())
        rows_delta = final_rows - raw_rows
        cols_delta = final_cols - raw_cols
        nulls_cleaned = nulls_before - nulls_after

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if have_raw:
            st.metric("📊 Rows (Original → Cleaned)", final_rows, delta=int(rows_delta))
        else:
            st.metric("📊 Cleaned Rows", final_rows)
    with c2:
        st.metric("🗑️ Duplicate Rows Removed", dup_count)
    with c3:
        if have_raw:
            st.metric("❌ Null Values Cleaned", nulls_cleaned)
        else:
            st.metric("❌ Nulls Remaining", nulls_after)
    with c4:
        if have_raw:
            st.metric("📐 Columns Kept", final_cols, delta=int(cols_delta))
        else:
            st.metric("📐 Columns Kept", final_cols)

    if not have_raw:
        st.caption(
            "ℹ️ A raw (pre-cleaning) baseline wasn't available for this file, "
            "so the metrics above show cleaned-output figures only."
        )


# --------------------------------------------------------------------------- #
# Interactive live filtering
# --------------------------------------------------------------------------- #

def apply_live_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    """
    Render the "🔍 Live Filter Options" sidebar section and return the
    filtered DataFrame. Uses df.map() (not the deprecated df.applymap())
    for the cell-level text search.
    """
    if not _is_nonempty_df(df):
        return df

    st.sidebar.markdown("### 🔍 Live Filter Options")

    nan_filter_cols = st.sidebar.multiselect(
        "Drop rows with NaN in:",
        options=list(df.columns),
        key=f"{key_prefix}_nan_filter",
        help="Rows containing NaN in ANY of the selected columns are removed.",
    )
    search_term = st.sidebar.text_input(
        "Search across all columns:",
        key=f"{key_prefix}_search",
        placeholder="Type to filter rows in real time…",
    )

    filtered = df
    if nan_filter_cols:
        filtered = filtered.dropna(subset=nan_filter_cols)

    if search_term:
        needle = search_term.lower()
        mask = filtered.map(
            lambda cell: needle in str(cell).lower() if pd.notna(cell) else False
        ).any(axis=1)
        filtered = filtered[mask]

    return filtered.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Batch processing: ZIP export helper
# --------------------------------------------------------------------------- #

def dataframes_to_zip_bytes(named_dataframes: Dict[str, pd.DataFrame], export_format: str = "csv") -> bytes:
    """
    Bundle multiple {name: DataFrame} pairs into a single in-memory ZIP
    archive, exporting each DataFrame as either CSV or Excel bytes
    (reusing dataframe_to_csv_bytes / dataframe_to_excel_bytes from
    converter.py for the actual per-file export).

    Args:
        named_dataframes: mapping of original filename -> cleaned DataFrame.
        export_format: "csv" or "excel". Defaults to "csv".

    Returns empty bytes() if there is nothing to zip or the archive
    build fails (never raises).
    """
    if not named_dataframes:
        return bytes()

    try:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, df in named_dataframes.items():
                if not _is_nonempty_df(df):
                    continue

                base_name = name.rsplit(".", 1)[0] if "." in name else name

                if export_format == "excel":
                    file_bytes = dataframe_to_excel_bytes(df)
                    out_name = f"{base_name}_cleaned.xlsx"
                else:
                    file_bytes = dataframe_to_csv_bytes(df)
                    out_name = f"{base_name}_cleaned.csv"

                if file_bytes:
                    zf.writestr(out_name, file_bytes)

        return buffer.getvalue()
    except Exception as e:
        st.error(f"Failed to build ZIP archive: {e}")
        return bytes()


# --------------------------------------------------------------------------- #
# Tab 1: Data Cleaners — PDF Cleaner
# --------------------------------------------------------------------------- #

def render_pdf_single() -> None:
    st.sidebar.markdown("### PDF Cleaner Options")
    force_ocr = st.sidebar.checkbox("Force OCR Mode", value=False, key="pdf_single_force_ocr")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="pdf_single_dedup")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_single_uploader")
    if uploaded_pdf is None:
        st.info("Upload a PDF file to get started.")
        return

    try:
        with st.spinner("Processing PDF…"):
            cleaned_no_dedup = process_pdf(uploaded_pdf, ocr_fallback=True, force_ocr=force_ocr)
            raw_df = None if force_ocr else _raw_read_pdf(uploaded_pdf)
    except Exception as e:
        st.error(f"Unexpected error while processing the PDF: {e}")
        return

    if not _is_nonempty_df(cleaned_no_dedup):
        st.warning(
            "No tabular data could be extracted from this PDF. "
            "Try enabling 'Force OCR Mode' in the sidebar if this is a scanned document."
        )
        return

    final_df, dup_count = _finalize(cleaned_no_dedup, drop_duplicates)
    render_summary_metrics(raw_df, dup_count, final_df)

    filtered_df = apply_live_filters(final_df, key_prefix="pdf_single")
    if not _is_nonempty_df(filtered_df):
        st.warning("No rows match the current filters.")
        return

    st.dataframe(filtered_df, use_container_width=True)

    csv_bytes = dataframe_to_csv_bytes(filtered_df)
    excel_bytes = dataframe_to_excel_bytes(filtered_df)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Cleaned CSV", data=csv_bytes, file_name="cleaned_pdf_data.csv",
            mime="text/csv", disabled=not csv_bytes, use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download Cleaned Excel", data=excel_bytes, file_name="cleaned_pdf_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not excel_bytes, use_container_width=True,
        )


def render_pdf_batch() -> None:
    st.sidebar.markdown("### PDF Cleaner Options (Batch)")
    force_ocr = st.sidebar.checkbox("Force OCR Mode", value=False, key="pdf_batch_force_ocr")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="pdf_batch_dedup")
    export_format = st.sidebar.radio("ZIP export format", ["CSV", "Excel"], key="pdf_batch_format")

    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True, key="pdf_batch_uploader"
    )
    if not uploaded_files:
        st.info("Upload one or more PDF files to run batch processing.")
        return

    results: Dict[str, pd.DataFrame] = {}
    progress_rows = []
    progress_bar = st.progress(0.0)

    for i, f in enumerate(uploaded_files):
        try:
            cleaned_no_dedup = process_pdf(f, ocr_fallback=True, force_ocr=force_ocr)
            if _is_nonempty_df(cleaned_no_dedup):
                final_df, dup_count = _finalize(cleaned_no_dedup, drop_duplicates)
                results[f.name] = final_df
                progress_rows.append({
                    "File": f.name, "Status": "✅ Success",
                    "Rows": final_df.shape[0], "Columns": final_df.shape[1],
                    "Duplicates Removed": dup_count,
                })
            else:
                progress_rows.append({
                    "File": f.name, "Status": "⚠️ No data extracted",
                    "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
                })
        except Exception as e:
            progress_rows.append({
                "File": f.name, "Status": f"❌ Error: {e}",
                "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
            })
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.dataframe(pd.DataFrame(progress_rows), use_container_width=True)

    if results:
        zip_bytes = dataframes_to_zip_bytes(results, export_format=export_format.lower())
        st.success(f"Processed {len(results)} of {len(uploaded_files)} files successfully.")
        st.download_button(
            "⬇️ Download All Cleaned Files (.ZIP)", data=zip_bytes,
            file_name="cleaned_pdfs.zip", mime="application/zip", disabled=not zip_bytes,
        )
    else:
        st.warning("No files were successfully processed.")


def render_pdf_cleaner_tab() -> None:
    st.subheader("📄 PDF Cleaner")
    st.caption("Extract structured tables from a digital or scanned PDF.")

    mode = st.radio(
        "Processing Mode", ["Single File Mode", "Batch Mode (Multiple Files)"],
        horizontal=True, key="pdf_mode",
    )
    st.divider()

    if mode == "Single File Mode":
        render_pdf_single()
    else:
        render_pdf_batch()


# --------------------------------------------------------------------------- #
# Tab 1: Data Cleaners — CSV Cleaner
# --------------------------------------------------------------------------- #

def render_csv_single() -> None:
    st.sidebar.markdown("### CSV Cleaner Options")
    delimiter_label = st.sidebar.selectbox("Delimiter", options=list(DELIMITER_OPTIONS.keys()), index=0, key="csv_single_delim")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="csv_single_dedup")

    uploaded_csv = st.file_uploader(
        "Upload a CSV, TSV, or TXT file", type=["csv", "tsv", "txt"], key="csv_single_uploader"
    )
    if uploaded_csv is None:
        st.info("Upload a CSV, TSV, or TXT file to get started.")
        return

    resolved_delimiter = DELIMITER_OPTIONS[delimiter_label]

    try:
        with st.spinner("Processing CSV…"):
            cleaned_no_dedup = process_csv(uploaded_csv, delimiter=resolved_delimiter, drop_duplicates=False)
            raw_df = _raw_read_csv(uploaded_csv, resolved_delimiter)
    except Exception as e:
        st.error(f"Unexpected error while processing the file: {e}")
        return

    if not _is_nonempty_df(cleaned_no_dedup):
        st.warning(
            "No usable data could be extracted from this file. "
            "Double-check the delimiter setting in the sidebar."
        )
        return

    final_df, dup_count = _finalize(cleaned_no_dedup, drop_duplicates)
    render_summary_metrics(raw_df, dup_count, final_df)

    filtered_df = apply_live_filters(final_df, key_prefix="csv_single")
    if not _is_nonempty_df(filtered_df):
        st.warning("No rows match the current filters.")
        return

    st.dataframe(filtered_df, use_container_width=True)

    csv_bytes = dataframe_to_csv_bytes(filtered_df)
    st.download_button(
        "⬇️ Download Cleaned CSV", data=csv_bytes, file_name="cleaned_data.csv",
        mime="text/csv", disabled=not csv_bytes,
    )


def render_csv_batch() -> None:
    st.sidebar.markdown("### CSV Cleaner Options (Batch)")
    delimiter_label = st.sidebar.selectbox("Delimiter", options=list(DELIMITER_OPTIONS.keys()), index=0, key="csv_batch_delim")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="csv_batch_dedup")
    export_format = st.sidebar.radio("ZIP export format", ["CSV", "Excel"], key="csv_batch_format")

    resolved_delimiter = DELIMITER_OPTIONS[delimiter_label]

    uploaded_files = st.file_uploader(
        "Upload CSV/TSV/TXT files", type=["csv", "tsv", "txt"],
        accept_multiple_files=True, key="csv_batch_uploader",
    )
    if not uploaded_files:
        st.info("Upload one or more files to run batch processing.")
        return

    results: Dict[str, pd.DataFrame] = {}
    progress_rows = []
    progress_bar = st.progress(0.0)

    for i, f in enumerate(uploaded_files):
        try:
            cleaned_no_dedup = process_csv(f, delimiter=resolved_delimiter, drop_duplicates=False)
            if _is_nonempty_df(cleaned_no_dedup):
                final_df, dup_count = _finalize(cleaned_no_dedup, drop_duplicates)
                results[f.name] = final_df
                progress_rows.append({
                    "File": f.name, "Status": "✅ Success",
                    "Rows": final_df.shape[0], "Columns": final_df.shape[1],
                    "Duplicates Removed": dup_count,
                })
            else:
                progress_rows.append({
                    "File": f.name, "Status": "⚠️ No data extracted",
                    "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
                })
        except Exception as e:
            progress_rows.append({
                "File": f.name, "Status": f"❌ Error: {e}",
                "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
            })
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.dataframe(pd.DataFrame(progress_rows), use_container_width=True)

    if results:
        zip_bytes = dataframes_to_zip_bytes(results, export_format=export_format.lower())
        st.success(f"Processed {len(results)} of {len(uploaded_files)} files successfully.")
        st.download_button(
            "⬇️ Download All Cleaned Files (.ZIP)", data=zip_bytes,
            file_name="cleaned_csvs.zip", mime="application/zip", disabled=not zip_bytes,
        )
    else:
        st.warning("No files were successfully processed.")


def render_csv_cleaner_tab() -> None:
    st.subheader("📑 CSV Cleaner")
    st.caption("Clean messy CSV/TSV/TXT files: encoding fixes, delimiter detection, NA normalization.")

    mode = st.radio(
        "Processing Mode", ["Single File Mode", "Batch Mode (Multiple Files)"],
        horizontal=True, key="csv_mode",
    )
    st.divider()

    if mode == "Single File Mode":
        render_csv_single()
    else:
        render_csv_batch()


# --------------------------------------------------------------------------- #
# Tab 1: Data Cleaners — Excel Cleaner
# --------------------------------------------------------------------------- #

def render_excel_single() -> None:
    st.sidebar.markdown("### Excel Cleaner Options")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="excel_single_dedup")

    uploaded_excel = st.file_uploader(
        "Upload an Excel workbook (.xlsx or .xls)", type=["xlsx", "xls"], key="excel_single_uploader"
    )
    if uploaded_excel is None:
        st.info("Upload an Excel workbook to get started.")
        return

    try:
        with st.spinner("Processing workbook…"):
            result = process_excel(uploaded_excel, sheet_name=None, drop_duplicates=False)
    except Exception as e:
        st.error(f"Unexpected error while processing the workbook: {e}")
        return

    if not isinstance(result, dict) or not result:
        st.warning("No usable data could be extracted from this workbook.")
        return

    non_empty_sheets = {name: df for name, df in result.items() if _is_nonempty_df(df)}
    if not non_empty_sheets:
        st.warning("This workbook was read, but every sheet is empty after cleaning.")
        return

    sheet_names = list(non_empty_sheets.keys())
    selected_sheet = st.selectbox("Select a sheet to preview", sheet_names, key="excel_single_sheet")
    selected_cleaned_no_dedup = non_empty_sheets[selected_sheet]

    raw_df = _raw_read_excel(uploaded_excel, selected_sheet)
    final_df, dup_count = _finalize(selected_cleaned_no_dedup, drop_duplicates)
    render_summary_metrics(raw_df, dup_count, final_df)

    filtered_df = apply_live_filters(final_df, key_prefix="excel_single")
    if not _is_nonempty_df(filtered_df):
        st.warning("No rows match the current filters.")
        return

    if len(non_empty_sheets) > 1:
        st.caption(
            f"This workbook has {len(non_empty_sheets)} non-empty sheets. "
            "The Excel download below includes all of them; the filters "
            "above apply only to the currently previewed sheet."
        )

    st.dataframe(filtered_df, use_container_width=True)

    # Build the export set: every sheet gets deduplicated per the sidebar
    # toggle, and the currently selected sheet is swapped for its filtered
    # version so the download reflects the active live filters.
    export_sheets: Dict[str, pd.DataFrame] = {}
    for name, sheet_df in non_empty_sheets.items():
        sheet_final, _ = _finalize(sheet_df, drop_duplicates)
        export_sheets[name] = sheet_final
    export_sheets[selected_sheet] = filtered_df

    excel_bytes = dataframe_to_excel_bytes(export_sheets)
    st.download_button(
        "⬇️ Download Cleaned Excel", data=excel_bytes, file_name="cleaned_workbook.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not excel_bytes,
    )


def render_excel_batch() -> None:
    st.sidebar.markdown("### Excel Cleaner Options (Batch)")
    drop_duplicates = st.sidebar.checkbox("Drop Duplicate Rows", value=True, key="excel_batch_dedup")
    export_format = st.sidebar.radio("ZIP export format", ["Excel", "CSV"], key="excel_batch_format")
    st.sidebar.caption("Batch mode processes each workbook's FIRST sheet only.")

    uploaded_files = st.file_uploader(
        "Upload Excel workbooks (.xlsx or .xls)", type=["xlsx", "xls"],
        accept_multiple_files=True, key="excel_batch_uploader",
    )
    if not uploaded_files:
        st.info("Upload one or more workbooks to run batch processing.")
        return

    results: Dict[str, pd.DataFrame] = {}
    progress_rows = []
    progress_bar = st.progress(0.0)

    for i, f in enumerate(uploaded_files):
        try:
            cleaned_no_dedup = process_excel(f, sheet_name=0, drop_duplicates=False)
            if _is_nonempty_df(cleaned_no_dedup):
                final_df, dup_count = _finalize(cleaned_no_dedup, drop_duplicates)
                results[f.name] = final_df
                progress_rows.append({
                    "File": f.name, "Status": "✅ Success",
                    "Rows": final_df.shape[0], "Columns": final_df.shape[1],
                    "Duplicates Removed": dup_count,
                })
            else:
                progress_rows.append({
                    "File": f.name, "Status": "⚠️ No data extracted",
                    "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
                })
        except Exception as e:
            progress_rows.append({
                "File": f.name, "Status": f"❌ Error: {e}",
                "Rows": 0, "Columns": 0, "Duplicates Removed": 0,
            })
        progress_bar.progress((i + 1) / len(uploaded_files))

    st.dataframe(pd.DataFrame(progress_rows), use_container_width=True)

    if results:
        zip_bytes = dataframes_to_zip_bytes(results, export_format=export_format.lower())
        st.success(f"Processed {len(results)} of {len(uploaded_files)} files successfully.")
        st.download_button(
            "⬇️ Download All Cleaned Files (.ZIP)", data=zip_bytes,
            file_name="cleaned_workbooks.zip", mime="application/zip", disabled=not zip_bytes,
        )
    else:
        st.warning("No files were successfully processed.")


def render_excel_cleaner_tab() -> None:
    st.subheader("📊 Excel Cleaner")
    st.caption("Clean .xlsx / .xls workbooks: whitespace, NA normalization, empty rows/cols, duplicates.")

    mode = st.radio(
        "Processing Mode", ["Single File Mode", "Batch Mode (Multiple Files)"],
        horizontal=True, key="excel_mode",
    )
    st.divider()

    if mode == "Single File Mode":
        render_excel_single()
    else:
        render_excel_batch()


def render_data_cleaners_tab() -> None:
    """Top-level router for the 'Data Cleaners' tab."""
    st.header("🧹 Data Cleaners")

    cleaner_choice = st.selectbox(
        "Choose a cleaning module", options=["PDF Cleaner", "CSV Cleaner", "Excel Cleaner"],
    )
    st.divider()

    if cleaner_choice == "PDF Cleaner":
        render_pdf_cleaner_tab()
    elif cleaner_choice == "CSV Cleaner":
        render_csv_cleaner_tab()
    elif cleaner_choice == "Excel Cleaner":
        render_excel_cleaner_tab()


# --------------------------------------------------------------------------- #
# Tab 2: File Converters
# --------------------------------------------------------------------------- #

CONVERSION_ROUTES = {
    "PDF to CSV": {
        "extensions": ["pdf"], "output_filename": "converted.csv", "mime": "text/csv",
    },
    "PDF to Excel": {
        "extensions": ["pdf"], "output_filename": "converted.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "Excel to CSV": {
        "extensions": ["xlsx", "xls"], "output_filename": "converted.csv", "mime": "text/csv",
    },
    "CSV to Excel": {
        "extensions": ["csv", "tsv", "txt"], "output_filename": "converted.xlsx",
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
}


def _run_conversion(route: str, uploaded_file) -> bytes:
    try:
        if route == "PDF to CSV":
            return convert_pdf_to_csv(uploaded_file)
        elif route == "PDF to Excel":
            return convert_pdf_to_excel(uploaded_file)
        elif route == "Excel to CSV":
            return convert_excel_to_csv(uploaded_file, sheet_name=0)
        elif route == "CSV to Excel":
            return convert_csv_to_excel(uploaded_file)
        else:
            return bytes()
    except Exception as e:
        st.error(f"Unexpected error during conversion: {e}")
        return bytes()


def render_file_converters_tab() -> None:
    st.header("🔄 File Converters")

    route = st.selectbox("Choose a conversion route", options=list(CONVERSION_ROUTES.keys()))
    route_config = CONVERSION_ROUTES[route]
    st.divider()

    if route == "Excel to CSV":
        st.caption(
            "Note: CSV files can't hold multiple sheets, so only the "
            "**first sheet** of the workbook is converted."
        )

    uploaded_file = st.file_uploader(
        f"Upload the input file for '{route}'",
        type=route_config["extensions"], key=f"converter_uploader_{route}",
    )
    if uploaded_file is None:
        st.info("Upload a file to run this conversion.")
        return

    with st.spinner(f"Converting ({route})…"):
        output_bytes = _run_conversion(route, uploaded_file)

    if output_bytes:
        st.success("Conversion complete — your file is ready to download.")
    else:
        st.warning(
            "Conversion produced no output. The source file may be empty, "
            "corrupted, or contain no extractable data."
        )

    if output_bytes:
        st.download_button(
            f"⬇️ Download {route_config['output_filename']}", data=output_bytes,
            file_name=route_config["output_filename"], mime=route_config["mime"],
        )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    st.title("🛠️ Data Cleaner & Converter Suite")
    st.caption("Clean and convert messy PDF, CSV, and Excel data — all in one place.")

    tab_cleaners, tab_converters = st.tabs(["🧹 Data Cleaners", "🔄 File Converters"])

    with tab_cleaners:
        render_data_cleaners_tab()

    with tab_converters:
        render_file_converters_tab()


if __name__ == "__main__":
    main()