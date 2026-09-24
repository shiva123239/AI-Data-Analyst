from pathlib import Path
import pandas as pd


def load_dataset(uploaded_file):
    """
    Load a CSV or XLSX file into a Pandas DataFrame.

    Supports:
    - Streamlit UploadedFile objects
    - Local file paths
    """

    # Handle Streamlit UploadedFile
    if hasattr(uploaded_file, "name"):
        file_name = uploaded_file.name.lower()

    # Handle local file path
    else:
        file_name = str(uploaded_file).lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a CSV or XLSX file."
        )

    return df