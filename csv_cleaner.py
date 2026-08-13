"""
csv_cleaner.py
==============

Backend processing module for the Data Cleaner & Converter suite.

This module contains pure Python functions for reading and cleaning
CSV-like tabular data. It has no hard dependency on a running Streamlit
runtime — type hints for `UploadedFile` are resolved lazily via
`TYPE_CHECKING` so this module can be imported and unit tested outside
of Streamlit.

Public API:
    process_csv(uploaded_file, delimiter=None, drop_duplicates=True,
                na_values=None) -> pd.DataFrame

Typical usage in app.py:
    from csv_cleaner import process_csv

    uploaded_file = st.file_uploader("Upload CSV", type=["csv", "tsv", "txt"])
    if uploaded_file is not None:
        df = process_csv(uploaded_file)
        st.dataframe(df)
"""

from __future__ import annotations

import csv
import io
import logging
from typing import List, Optional, Sequence, TYPE_CHECKING

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

# Encodings to try, in order, when decoding raw bytes.
CANDIDATE_ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

# Delimiters to consider during auto-detection.
CANDIDATE_DELIMITERS: List[str] = [",", ";", "\t", "|"]

# String tokens that should be treated as missing values.
DEFAULT_NA_TOKENS: List[str] = [
    "", "N/A", "n/a", "NA", "na", "null", "NULL", "Null",
    "None", "none", "NONE", "-", "--", "?",
]


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

    # `read()` may return str for text-mode streams; normalize to bytes.
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def _decode_bytes(raw_bytes: bytes) -> Optional[str]:
    """
    Attempt to decode raw bytes into text, trying a sequence of common
    encodings. Returns None if every attempt fails.
    """
    for encoding in CANDIDATE_ENCODINGS:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    logger.warning(
        "Failed to decode file with any of the candidate encodings: %s",
        CANDIDATE_ENCODINGS,
    )
    return None


def _detect_delimiter(sample_text: str) -> str:
    """
    Auto-detect the most likely delimiter in a text sample.

    Uses `csv.Sniffer` first (checks against our candidate delimiter set),
    falling back to a simple frequency count of candidate delimiters in the
    first few lines if the Sniffer can't make a confident guess.
    """
    sample = "\n".join(sample_text.splitlines()[:20])  # first 20 lines is plenty

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS))
        if dialect.delimiter in CANDIDATE_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass  # Sniffer couldn't determine a dialect; fall back below.

    # Fallback: count occurrences of each candidate delimiter and pick the
    # most frequent one across the sample lines.
    counts = {delim: sample.count(delim) for delim in CANDIDATE_DELIMITERS}
    best_delim = max(counts, key=counts.get)

    if counts[best_delim] == 0:
        # No candidate delimiter found at all; default to comma.
        return ","

    return best_delim


def _clean_cell(value):
    """
    Strip whitespace from a single cell value if it's a string.
    Non-string values (numbers, NaN, etc.) are returned unchanged.
    """
    if isinstance(value, str):
        stripped = value.strip()
        return stripped
    return value


def _clean_dataframe(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    na_values: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Apply the standard cleaning pipeline to a raw-loaded DataFrame:
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

def process_csv(
    uploaded_file,
    delimiter: Optional[str] = None,
    drop_duplicates: bool = True,
    na_values: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Main entry point: process an uploaded CSV/TSV-like file into a cleaned
    pandas DataFrame.

    Args:
        uploaded_file: A Streamlit UploadedFile object, any file-like
            object supporting `.read()`, or raw bytes.
        delimiter: Optional explicit delimiter (",", ";", "\\t", "|", or
            any custom single-character string). If None, the delimiter
            is auto-detected.
        drop_duplicates: If True (default), remove fully duplicate rows
            after cleaning.
        na_values: Optional extra strings to treat as missing values, in
            addition to the built-in defaults (N/A, null, None, "", etc.).

    Returns:
        A cleaned pandas DataFrame. If reading or parsing fails for any
        reason (empty file, undecodable bytes, malformed CSV structure),
        an empty DataFrame is returned rather than raising.
    """
    if uploaded_file is None:
        logger.warning("process_csv called with no file.")
        return pd.DataFrame()

    # --- Step 1: Read raw bytes ---
    try:
        raw_bytes = _read_uploaded_file_bytes(uploaded_file)
    except Exception as e:
        logger.error("Failed to read uploaded file bytes: %s", e)
        return pd.DataFrame()

    if not raw_bytes:
        logger.warning("Uploaded file is empty.")
        return pd.DataFrame()

    # --- Step 2: Decode bytes to text (with encoding fallback) ---
    try:
        text = _decode_bytes(raw_bytes)
    except Exception as e:
        logger.error("Unexpected error while decoding file: %s", e)
        text = None

    if text is None or not text.strip():
        logger.warning("Could not decode file content or file is blank.")
        return pd.DataFrame()

    # --- Step 3: Determine delimiter ---
    try:
        resolved_delimiter = delimiter if delimiter else _detect_delimiter(text)
    except Exception as e:
        logger.warning("Delimiter detection failed, defaulting to comma: %s", e)
        resolved_delimiter = ","

    # --- Step 4: Parse into a DataFrame ---
    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=resolved_delimiter,
            engine="python",       # more tolerant of irregular rows
            skip_blank_lines=True,
            on_bad_lines="skip",   # skip malformed rows rather than raising
        )
    except Exception as e:
        logger.error("pandas failed to parse CSV content: %s", e)
        return pd.DataFrame()

    # --- Step 5: Clean the parsed DataFrame ---
    try:
        cleaned_df = _clean_dataframe(
            df,
            drop_duplicates=drop_duplicates,
            na_values=na_values,
        )
    except Exception as e:
        logger.error("Unexpected error during DataFrame cleaning: %s", e)
        return pd.DataFrame()

    return cleaned_df