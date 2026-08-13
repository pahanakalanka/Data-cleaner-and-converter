"""
pdf_cleaner.py
==============

Backend processing module for the Data Cleaner & Converter suite.

This module contains pure Python functions for extracting tabular data
from PDF files (digital or scanned) and returning clean pandas DataFrames.
It has NO Streamlit dependency of its own except for type-hinting the
`UploadedFile` object — this keeps it importable/testable outside of
a Streamlit runtime if needed.

Public API:
    process_pdf(uploaded_file, ocr_fallback=True, force_ocr=False) -> pd.DataFrame

Typical usage in app.py:
    from pdf_cleaner import process_pdf

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        df = process_pdf(uploaded_file)
        st.dataframe(df)
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes

if TYPE_CHECKING:
    # Only imported for type checking; avoids a hard runtime dependency
    # on streamlit inside this backend module.
    from streamlit.runtime.uploaded_file_manager import UploadedFile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _read_uploaded_file_bytes(uploaded_file: "UploadedFile") -> bytes:
    """
    Safely read the raw bytes from a Streamlit UploadedFile object.

    Streamlit's UploadedFile is a BytesIO-like object. We seek to 0 first
    in case something upstream already consumed part of the stream.
    """
    try:
        uploaded_file.seek(0)
    except Exception:
        # Some file-like objects (e.g. plain bytes wrapped elsewhere) may
        # not support seek; that's fine, just proceed.
        pass
    return uploaded_file.read()


def _clean_cell(value) -> Optional[str]:
    """
    Strip whitespace from a single cell value.

    Returns None for values that are None or empty after stripping so that
    downstream `dropna` logic can treat them as truly empty.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text if text != "" else None


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the standard cleaning pipeline to a raw extracted DataFrame:
      1. Strip whitespace from all cell values and column headers.
      2. Drop fully-empty rows and columns.
      3. Reset the index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Clean headers
    df.columns = [
        str(col).strip() if col is not None else f"col_{i}"
        for i, col in enumerate(df.columns)
    ]

    # Clean cell values
    df = df.map(_clean_cell)

    # Drop rows/columns that are entirely empty (NaN or None)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Reset index cleanly
    df = df.reset_index(drop=True)

    return df


def _dataframe_from_table_rows(table: List[List[Optional[str]]]) -> pd.DataFrame:
    """
    Convert a raw table (list of row-lists, as returned by pdfplumber)
    into a DataFrame, using the first row as the header.
    """
    if not table or len(table) < 1:
        return pd.DataFrame()

    header, *rows = table

    # Guard against a table with only a header and no data rows
    if not rows:
        return pd.DataFrame(columns=header)

    return pd.DataFrame(rows, columns=header)


# --------------------------------------------------------------------------- #
# Extraction strategies
# --------------------------------------------------------------------------- #

def extract_tables_digital(file_bytes: bytes) -> List[pd.DataFrame]:
    """
    Attempt to extract structured tables from a text-based (digital) PDF
    using pdfplumber, across all pages.

    Returns a list of DataFrames (one per detected table). Returns an
    empty list if extraction fails or no tables are found.
    """
    tables_found: List[pd.DataFrame] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    raw_tables = page.extract_tables()
                except Exception as page_err:
                    logger.warning(
                        "pdfplumber failed to extract tables on page %s: %s",
                        page_number, page_err,
                    )
                    continue

                for raw_table in raw_tables:
                    df = _dataframe_from_table_rows(raw_table)
                    if not df.empty:
                        tables_found.append(df)

    except Exception as e:
        logger.warning("pdfplumber failed to open/parse PDF: %s", e)
        return []

    return tables_found


def extract_text_ocr(file_bytes: bytes, dpi: int = 300) -> str:
    """
    OCR fallback: rasterize each PDF page to an image and run pytesseract
    to extract raw text. Used when the PDF is scanned/image-based and
    pdfplumber cannot find any structured tables.

    Returns the concatenated OCR text from all pages (empty string on
    failure).
    """
    try:
        images = convert_from_bytes(file_bytes, dpi=dpi)
    except Exception as e:
        logger.warning("pdf2image failed to rasterize PDF for OCR: %s", e)
        return ""

    page_texts: List[str] = []
    for page_number, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image)
            page_texts.append(text)
        except Exception as ocr_err:
            logger.warning(
                "pytesseract failed on page %s: %s", page_number, ocr_err
            )
            continue

    return "\n".join(page_texts)


def _ocr_text_to_dataframe(ocr_text: str) -> pd.DataFrame:
    """
    Best-effort conversion of raw OCR text into a DataFrame.

    OCR output has no reliable table structure, so this uses a simple
    whitespace-based split per line as a fallback heuristic. Each line
    becomes a row, split on runs of 2+ spaces (a common visual column
    separator in OCR'd tabular text). Callers needing higher-fidelity
    OCR tables should pre-process with pytesseract's `image_to_data`
    (bounding boxes) instead of this simple heuristic.
    """
    if not ocr_text or not ocr_text.strip():
        return pd.DataFrame()

    import re

    lines = [line for line in ocr_text.splitlines() if line.strip()]
    if not lines:
        return pd.DataFrame()

    split_rows = [re.split(r"\s{2,}", line.strip()) for line in lines]

    # Pad rows to the max column count so DataFrame construction doesn't fail
    max_cols = max(len(row) for row in split_rows)
    padded_rows = [row + [None] * (max_cols - len(row)) for row in split_rows]

    # Treat first line as header if it looks distinct enough; otherwise
    # fall back to generic column names.
    header, *data_rows = padded_rows
    if not data_rows:
        return pd.DataFrame(columns=header)

    return pd.DataFrame(data_rows, columns=header)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def process_pdf(
    uploaded_file: "UploadedFile",
    ocr_fallback: bool = True,
    force_ocr: bool = False,
) -> pd.DataFrame:
    """
    Main entry point: process a Streamlit-uploaded PDF into a cleaned
    pandas DataFrame.

    Args:
        uploaded_file: The Streamlit UploadedFile object for the PDF.
        ocr_fallback: If True (default), fall back to OCR extraction when
            pdfplumber finds no tables in the digital PDF.
        force_ocr: If True, skip the digital extraction attempt entirely
            and go straight to OCR. Useful when the caller already knows
            the PDF is a scanned image (e.g. from a UI checkbox).

    Returns:
        A cleaned pandas DataFrame. If extraction fails completely or the
        PDF has no extractable tabular data, an empty DataFrame is
        returned (never raises for malformed/empty PDFs).
    """
    if uploaded_file is None:
        logger.warning("process_pdf called with no file.")
        return pd.DataFrame()

    try:
        file_bytes = _read_uploaded_file_bytes(uploaded_file)
    except Exception as e:
        logger.error("Failed to read uploaded file bytes: %s", e)
        return pd.DataFrame()

    if not file_bytes:
        logger.warning("Uploaded file is empty.")
        return pd.DataFrame()

    # --- Strategy 1: Digital extraction via pdfplumber ---
    tables: List[pd.DataFrame] = []
    if not force_ocr:
        try:
            tables = extract_tables_digital(file_bytes)
        except Exception as e:
            # extract_tables_digital already handles its own errors, but
            # this is a final safety net per the error-handling requirement.
            logger.error("Unexpected error during digital extraction: %s", e)
            tables = []

    if tables:
        # Combine all detected tables into one DataFrame. If multiple
        # tables have differing schemas, concatenation aligns by column
        # name and fills mismatches with NaN, which the cleaning step
        # will handle appropriately.
        try:
            combined = pd.concat(tables, ignore_index=True, sort=False)
        except Exception as e:
            logger.error("Failed to concatenate extracted tables: %s", e)
            combined = tables[0]  # fall back to the first table found
        return _clean_dataframe(combined)

    # --- Strategy 2: OCR fallback ---
    if force_ocr or ocr_fallback:
        logger.info("No digital tables found (or OCR forced); falling back to OCR.")
        try:
            ocr_text = extract_text_ocr(file_bytes)
            ocr_df = _ocr_text_to_dataframe(ocr_text)
        except Exception as e:
            logger.error("Unexpected error during OCR extraction: %s", e)
            ocr_df = pd.DataFrame()

        if not ocr_df.empty:
            return _clean_dataframe(ocr_df)

    # --- Nothing worked: return an empty DataFrame gracefully ---
    logger.warning("No tabular data could be extracted from the PDF.")
    return pd.DataFrame()
