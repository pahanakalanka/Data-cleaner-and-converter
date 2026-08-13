"""
excel_cleaner.py
=================

Backend processing module for the Data Cleaner & Converter suite.

This module contains pure Python functions for reading and cleaning
Excel workbook data (.xlsx via openpyxl, legacy .xls via xlrd). It has no
hard dependency on a running Streamlit runtime — type hints for
`UploadedFile` are resolved lazily via `TYPE_CHECKING` so this module can
be imported and unit tested outside of Streamlit.

Public API:
    process_excel(uploaded_file, sheet_name=0, drop_duplicates=True,
                   na_values=None) -> pd.DataFrame | dict[str, pd.DataFrame]

Typical usage in app.py:
    from excel_cleaner import process_excel

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        result = process_excel(uploaded_file)
        if isinstance(result, dict):
            sheet = st.selectbox("Sheet", list(result.keys()))
            st.dataframe(result[sheet])
        else:
            st.dataframe(result)
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Only imported for type checking; avoids a hard runtime dependency
    # on streamlit inside this backend module.
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# String tokens that should be treated as missing values.
DEFAULT_NA_TOKENS: List[str] = [
    "", "N/A", "n/a", "NA", "na", "null", "NULL", "Null",
    "None", "none", "NONE", "-", "--", "?",
]

# File extensions mapped to the pandas/openpyxl "engine" argument.
ENGINE_BY_EXTENSION = {
    "xlsx": "openpyxl",
    "xlsm": "openpyxl",
    "xls": "xlrd",
}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _read_uploaded_file_bytes(uploaded_file) -> bytes:
    """
    Safely read raw bytes from a Streamlit UploadedFile object, a plain
    file-like stream, or raw bytes.

    Seeks to 0 first (when supported) in case something upstream already
    consumed part of the stream.
    """
    if isinstance(uploaded_file, (bytes, bytearray)):
        return bytes(uploaded_file)

    try:
        uploaded_file.seek(0)
    except Exception:
        # Some file-like objects may not support seek; proceed anyway.
        pass

    data = uploaded_file.read()

    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _guess_engine(uploaded_file, raw_bytes: bytes) -> Optional[str]:
    """
    Determine which pandas engine to use ("openpyxl" for .xlsx/.xlsm,
    "xlrd" for legacy .xls), based on the uploaded file's name if
    available, falling back to sniffing the byte signature.

    Returns None if the engine cannot be confidently determined, in which
    case pandas will be left to auto-detect (which works for most cases
    but can fail on ambiguous or misnamed files).
    """
    filename = getattr(uploaded_file, "name", None)
    if filename and "." in filename:
        extension = filename.rsplit(".", 1)[-1].lower()
        if extension in ENGINE_BY_EXTENSION:
            return ENGINE_BY_EXTENSION[extension]

    # Fallback: sniff the byte signature.
    # .xlsx/.xlsm files are ZIP archives -> start with "PK\x03\x04".
    # Legacy .xls files are OLE2 compound documents -> start with
    # "\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1".
    if raw_bytes[:4] == b"PK\x03\x04":
        return "openpyxl"
    if raw_bytes[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1":
        return "xlrd"

    return None


def _clean_cell(value):
    """
    Strip whitespace from a single cell value if it's a string.
    Non-string values (numbers, dates, NaN, etc.) are returned unchanged.
    """
    if isinstance(value, str):
        return value.strip()
    return value


def _clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    na_values: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Apply the standard cleaning pipeline to a single raw-loaded sheet
    DataFrame:
      1. Strip whitespace from column headers and string cell values.
      2. Normalize common "missing value" string tokens (N/A, null, etc.)
         into proper NaN.
      3. Drop fully-empty rows and columns.
      4. Optionally drop duplicate rows.
      5. Reset the index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Clean headers
    df.columns = [
        str(col).strip() if col is not None else f"col_{i}"
        for i, col in enumerate(df.columns)
    ]

    # 1b. Clean string cell values
    df = df.map(_clean_cell)

    # 2. Normalize missing-value tokens to NaN
    tokens_to_treat_as_na = set(DEFAULT_NA_TOKENS)
    if na_values:
        tokens_to_treat_as_na.update(na_values)
    df = df.replace(list(tokens_to_treat_as_na), np.nan)

    # 3. Drop fully-empty rows and columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # 4. Optionally drop duplicate rows
    if drop_duplicates:
        df = df.drop_duplicates()

    # 5. Reset index cleanly
    df = df.reset_index(drop=True)

    return df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def process_excel(
    uploaded_file,
    sheet_name: Union[str, int, None] = 0,
    drop_duplicates: bool = True,
    na_values: Optional[Sequence[str]] = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Main entry point: process an uploaded Excel workbook into a cleaned
    pandas DataFrame (single sheet) or a dict of cleaned DataFrames
    (multiple sheets).

    Args:
        uploaded_file: A Streamlit UploadedFile object, any file-like
            object supporting `.read()`, or raw bytes.
        sheet_name: Which sheet(s) to read.
            - int (default 0): read the sheet at that positional index.
            - str: read the sheet with that exact name.
            - None: read ALL sheets, returning a dict of
              {sheet_name: cleaned_DataFrame}.
        drop_duplicates: If True (default), remove fully duplicate rows
            after cleaning, applied per-sheet.
        na_values: Optional extra strings to treat as missing values, in
            addition to the built-in defaults (N/A, null, None, "", etc.).

    Returns:
        A single cleaned pandas DataFrame if `sheet_name` is an int or
        str, or a dict of {sheet_name: DataFrame} if `sheet_name` is
        None. On any failure (corrupted workbook, unreadable bytes,
        unsupported format), returns an empty DataFrame (or an empty
        dict, when `sheet_name=None`) rather than raising.
    """
    empty_result: Union[pd.DataFrame, Dict[str, pd.DataFrame]] = (
        {} if sheet_name is None else pd.DataFrame()
    )

    if uploaded_file is None:
        logger.warning("process_excel called with no file.")
        return empty_result

    # --- Step 1: Read raw bytes ---
    try:
        raw_bytes = _read_uploaded_file_bytes(uploaded_file)
    except Exception as e:
        logger.error("Failed to read uploaded file bytes: %s", e)
        return empty_result

    if not raw_bytes:
        logger.warning("Uploaded file is empty.")
        return empty_result

    # --- Step 2: Determine the correct engine (openpyxl vs xlrd) ---
    engine = _guess_engine(uploaded_file, raw_bytes)

    # --- Step 3: Open the workbook ---
    try:
        excel_file = pd.ExcelFile(io.BytesIO(raw_bytes), engine=engine)
    except Exception as e:
        logger.error("Failed to open workbook (corrupted or unsupported format): %s", e)
        return empty_result

    # --- Step 4: Read requested sheet(s) ---
    try:
        raw_data = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            engine=engine,
        )
    except Exception as e:
        logger.error("Failed to read sheet(s) '%s' from workbook: %s", sheet_name, e)
        return empty_result

    # --- Step 5: Clean the result ---
    # pandas returns a dict when sheet_name=None (all sheets), and a
    # single DataFrame otherwise.
    try:
        if isinstance(raw_data, dict):
            cleaned: Dict[str, pd.DataFrame] = {}
            for name, sheet_df in raw_data.items():
                try:
                    cleaned[name] = _clean_dataframe(
                        sheet_df,
                        drop_duplicates=drop_duplicates,
                        na_values=na_values,
                    )
                except Exception as sheet_err:
                    logger.warning(
                        "Failed to clean sheet '%s', skipping: %s", name, sheet_err
                    )
                    cleaned[name] = pd.DataFrame()
            return cleaned
        else:
            return _clean_dataframe(
                raw_data,
                drop_duplicates=drop_duplicates,
                na_values=na_values,
            )
    except Exception as e:
        logger.error("Unexpected error during DataFrame cleaning: %s", e)
        return empty_result