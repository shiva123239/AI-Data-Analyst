"""
RAG Data Ingestion

Converts tabular data into LangChain Documents
and stores them in the RAG knowledge base.
"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.ingestion.file_loader import load_dataset
from src.ai.rag_engine import RAGEngine


def dataframe_to_documents(
    df: pd.DataFrame,
    source: str = "dataset",
) -> list[Document]:
    """
    Convert each DataFrame row into a LangChain Document.

    Each row becomes one searchable document.
    """

    documents = []

    for index, row in df.iterrows():

        row_data = []

        for column in df.columns:
            value = row[column]

            if pd.isna(value):
                value = ""

            row_data.append(f"{column}: {value}")

        content = "\n".join(row_data)

        document = Document(
            page_content=content,
            metadata={
                "source": source,
                "row": int(index),
            },
        )

        documents.append(document)

    return documents


def ingest_file(
    file_path: str,
    rag_engine: RAGEngine | None = None,
) -> int:
    """
    Load a CSV/XLSX file and add its rows to the RAG knowledge base.

    Returns:
        Number of documents added.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"Loading dataset: {path}")

    df = load_dataset(str(path))

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    documents = dataframe_to_documents(
        df,
        source=path.name,
    )

    print(f"Created {len(documents)} documents.")

    if rag_engine is None:
        rag_engine = RAGEngine()

    chunks_added = rag_engine.add_documents(documents)

    print(f"Added {chunks_added} chunks to RAG knowledge base.")

    return chunks_added


if __name__ == "__main__":

    DATASET_PATH = "data/sales_data.csv"

    print("=" * 60)
    print("RAG DATA INGESTION")
    print("=" * 60)

    ingest_file(DATASET_PATH)

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)