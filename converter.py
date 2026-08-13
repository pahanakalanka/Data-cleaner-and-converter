"""
converter.py
============

Backend processing module for the Data Cleaner & Converter suite.

This module is responsible for cross-format conversion (PDF -> CSV/Excel,
Excel -> CSV, CSV -> Excel). It does NOT re-implement any parsing or
cleaning logic itself — it reuses the existing `process_pdf`,
`process_csv`, and `process_excel` functions from the respective cleaner
modules to turn an uploaded file into a cleaned DataFrame, then exports
that DataFrame into the target format's raw bytes.

It has no hard dependency on a running Streamlit runtime — type hints for
`UploadedFile` are resolved lazily via `TYPE_CHECKING` so this module can
be imported and unit tested outside of Streamlit.

Public API:
    dataframe_to_csv_bytes(df) -> bytes
    dataframe_to_excel_bytes(df_or_dict) -> bytes
    dataframes_to_zip_bytes(files_dict, output_format) -> bytes

    convert_pdf_to_csv(uploaded_file) -> bytes
    convert_pdf_to_excel(uploaded_file) -> bytes
    convert_excel_to_csv(uploaded_file, sheet_name=0) -> bytes
    convert_csv_to_excel(uploaded_file) -> bytes
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Dict, TYPE_CHECKING, Union

import pandas as pd

from pdf_cleaner import process_pdf
from csv_cleaner import process_csv
from excel_cleaner import process_excel

if TYPE_CHECKING:
    # Only imported for type checking; avoids a hard runtime dependency
    # on streamlit inside this backend module.
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export a single DataFrame to raw CSV bytes (UTF-8 encoded, no index
    column).

    Returns an empty bytes object if `df` is None/empty or if export
    fails for any reason (never raises).
    """
    if df is None or df.empty:
        logger.warning("dataframe_to_csv_bytes called with an empty/None DataFrame.")
        return bytes()

    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
    except Exception as e:
        logger.error("Failed to export DataFrame to CSV bytes: %s", e)
        return bytes()


def dataframe_to_excel_bytes(
    df_or_dict: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
) -> bytes:
    """
    Export a single DataFrame, or a dict of {sheet_name: DataFrame}, to
    an in-memory Excel (.xlsx) file using openpyxl.

    - If given a single DataFrame, it is written to a sheet named "Sheet1".
    - If given a dict, each entry becomes its own sheet, named after its
      key (truncated to Excel's 31-character sheet-name limit, since
      Excel will otherwise reject longer names).
    - Empty/None input, or an all-empty dict, returns empty bytes.

    Returns an empty bytes object if export fails for any reason (never
    raises).
    """
    sheets: Dict[str, pd.DataFrame] = {}

    if isinstance(df_or_dict, dict):
        for name, sheet_df in df_or_dict.items():
            if sheet_df is not None and not sheet_df.empty:
                safe_name = str(name)[:31] if name else "Sheet1"
                sheets[safe_name] = sheet_df
    elif isinstance(df_or_dict, pd.DataFrame):
        if not df_or_dict.empty:
            sheets["Sheet1"] = df_or_dict
    else:
        logger.warning(
            "dataframe_to_excel_bytes called with unsupported type: %s",
            type(df_or_dict),
        )
        return bytes()

    if not sheets:
        logger.warning("dataframe_to_excel_bytes: no non-empty sheets to write.")
        return bytes()

    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for sheet_name, sheet_df in sheets.items():
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        return buffer.getvalue()
    except Exception as e:
        logger.error("Failed to export DataFrame(s) to Excel bytes: %s", e)
        return bytes()


def dataframes_to_zip_bytes(
    files_dict: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
    output_format: str = "csv",
) -> bytes:
    """
    Take a dict of {filename: DataFrame or sheet_dict} and package all cleaned 
    outputs into a single in-memory ZIP archive.
    
    `output_format`: "csv" or "excel" (or "xlsx")
    """
    if not files_dict:
        return bytes()

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, data in files_dict.items():
                base_name = name.rsplit(".", 1)[0] if "." in name else name

                if output_format.lower() in ["excel", "xlsx"]:
                    content = dataframe_to_excel_bytes(data)
                    if content:
                        zip_file.writestr(f"{base_name}_cleaned.xlsx", content)
                else:
                    # If data is a dict of sheets (from Excel), extract the first sheet
                    df = list(data.values())[0] if isinstance(data, dict) else data
                    content = dataframe_to_csv_bytes(df)
                    if content:
                        zip_file.writestr(f"{base_name}_cleaned.csv", content)

        return zip_buffer.getvalue()
    except Exception as e:
        logger.error("Failed to generate ZIP archive: %s", e)
        return bytes()


# --------------------------------------------------------------------------- #
# Core public conversion functions
# --------------------------------------------------------------------------- #

def convert_pdf_to_csv(uploaded_file) -> bytes:
    """Convert an uploaded PDF file into CSV bytes."""
    try:
        df = process_pdf(uploaded_file)
    except Exception as e:
        logger.error("convert_pdf_to_csv: PDF processing failed: %s", e)
        return bytes()

    if df is None or df.empty:
        logger.warning("convert_pdf_to_csv: no data extracted from PDF.")
        return bytes()

    return dataframe_to_csv_bytes(df)


def convert_pdf_to_excel(uploaded_file) -> bytes:
    """Convert an uploaded PDF file into Excel (.xlsx) bytes."""
    try:
        df = process_pdf(uploaded_file)
    except Exception as e:
        logger.error("convert_pdf_to_excel: PDF processing failed: %s", e)
        return bytes()

    if df is None or df.empty:
        logger.warning("convert_pdf_to_excel: no data extracted from PDF.")
        return bytes()

    return dataframe_to_excel_bytes(df)


def convert_excel_to_csv(uploaded_file, sheet_name: Union[str, int, None] = 0) -> bytes:
    """Convert an uploaded Excel workbook into CSV bytes."""
    try:
        if sheet_name is None:
            logger.warning(
                "convert_excel_to_csv: sheet_name=None is not supported for a "
                "single CSV; falling back to the first sheet (index 0)."
            )
            result = process_excel(uploaded_file, sheet_name=0)
        else:
            result = process_excel(uploaded_file, sheet_name=sheet_name)
    except Exception as e:
        logger.error("convert_excel_to_csv: Excel processing failed: %s", e)
        return bytes()

    if isinstance(result, dict):
        first_sheet_df = next(iter(result.values()), pd.DataFrame())
        df = first_sheet_df
    else:
        df = result

    if df is None or df.empty:
        logger.warning("convert_excel_to_csv: no data extracted from workbook.")
        return bytes()

    return dataframe_to_csv_bytes(df)


def convert_csv_to_excel(uploaded_file) -> bytes:
    """Convert an uploaded CSV file into Excel (.xlsx) bytes."""
    try:
        df = process_csv(uploaded_file)
    except Exception as e:
        logger.error("convert_csv_to_excel: CSV processing failed: %s", e)
        return bytes()

    if df is None or df.empty:
        logger.warning("convert_csv_to_excel: no data extracted from CSV.")
        return bytes()

    return dataframe_to_excel_bytes(df)