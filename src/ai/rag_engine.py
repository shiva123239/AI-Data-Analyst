"""
RAG Engine
----------
Local Retrieval-Augmented Generation engine using:

- Ollama
- Llama 3.2
- nomic-embed-text
- ChromaDB
- LangChain

Also supports exact calculations on the structured sales dataset.
"""

from pathlib import Path
from typing import List
import time

import pandas as pd

from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGEngine:
    """Local RAG engine for the AI Data Analyst application."""

    def __init__(
        self,
        llm_model: str = "llama3.2",
        embedding_model: str = "nomic-embed-text",
        persist_directory: str = "models/rag_chroma",
        dataset_path: str = "data/sales_data.csv",
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.dataset_path = dataset_path

        # ============================================================
        # OLLAMA CONFIGURATION
        # ============================================================

        self.ollama_base_url = "http://127.0.0.1:11434"

        # ============================================================
        # LOCAL OLLAMA LLM
        # ============================================================

        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=0,
            base_url=self.ollama_base_url,
        )

        # ============================================================
        # LOCAL OLLAMA EMBEDDINGS
        # ============================================================

        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )

        # ============================================================
        # TEXT SPLITTER
        # ============================================================

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        # ============================================================
        # CHROMA VECTOR DATABASE
        # ============================================================

        self.vector_store = Chroma(
            collection_name="ai_data_analyst",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    # ================================================================
    # DOCUMENT MANAGEMENT
    # ================================================================

    def add_documents(
        self,
        documents: List[Document],
        max_retries: int = 3,
    ) -> int:
        """
        Split documents into chunks and add them to ChromaDB.

        IMPORTANT:
        Documents are inserted ONE CHUNK AT A TIME.

        This avoids sending a large batch of embedding requests
        to Ollama, which can cause internal Ollama runner/tokenize
        connection errors on Windows.

        Returns:
            Number of successfully added chunks.
        """

        if not documents:
            return 0

        # ------------------------------------------------------------
        # Split documents into smaller chunks
        # ------------------------------------------------------------

        chunks = self.text_splitter.split_documents(documents)

        if not chunks:
            return 0

        successful_chunks = 0

        # ------------------------------------------------------------
        # Add each chunk separately
        # ------------------------------------------------------------

        for index, chunk in enumerate(chunks):

            last_error = None

            for attempt in range(1, max_retries + 1):

                try:

                    # Add ONLY ONE document at a time.
                    self.vector_store.add_documents(
                        [chunk]
                    )

                    successful_chunks += 1

                    # Small pause prevents Ollama's internal
                    # runner from being overwhelmed.
                    time.sleep(0.05)

                    break

                except Exception as e:

                    last_error = e

                    # Wait before retrying.
                    time.sleep(
                        1.0 * attempt
                    )

            else:

                # If this particular chunk still fails after
                # all retries, raise the actual error.
                raise RuntimeError(
                    "Ollama embedding failed while processing "
                    f"chunk {index + 1}/{len(chunks)} "
                    f"after {max_retries} attempts.\n\n"
                    f"Original error:\n{last_error}"
                )

        return successful_chunks

    # ================================================================
    # ADD TEXT
    # ================================================================

    def add_text(
        self,
        text: str,
        source: str = "user_input",
    ) -> int:
        """Add plain text to the knowledge base."""

        if not text or not text.strip():
            return 0

        document = Document(
            page_content=text,
            metadata={
                "source": source
            },
        )

        return self.add_documents(
            [document]
        )

    # ================================================================
    # RETRIEVAL
    # ================================================================

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[Document]:
        """Retrieve the most relevant documents."""

        if not query or not query.strip():
            return []

        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    # ================================================================
    # LOAD STRUCTURED DATA
    # ================================================================

    def load_dataframe(self) -> pd.DataFrame:
        """Load the structured sales dataset."""

        path = Path(
            self.dataset_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        return pd.read_csv(
            path
        )

    # ================================================================
    # STRUCTURED DATA QUESTIONS
    # ================================================================

    def answer_data_question(
        self,
        question: str,
    ) -> str | None:
        """
        Answer questions that require exact calculations
        from the structured sales dataset.

        Returns:
            Answer string if this is a structured-data question.
            None if the question should be handled by RAG/LLM.
        """

        q = question.lower().strip()

        # ============================================================
        # TOTAL SALES
        # ============================================================

        if (
            "total sales" in q
            or "sales total" in q
            or "sum of sales" in q
        ):

            df = self.load_dataframe()

            total_sales = df["Sales"].sum()

            return (
                "The total sales in the dataset are "
                f"{total_sales:,.0f}."
            )

        # ============================================================
        # HIGHEST SALES REGION
        # ============================================================

        if (
            "region" in q
            and (
                "highest sales" in q
                or "most sales" in q
                or "best sales" in q
            )
        ):

            df = self.load_dataframe()

            region_sales = (
                df.groupby("Region")["Sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            highest_region = (
                region_sales.index[0]
            )

            highest_sales = (
                region_sales.iloc[0]
            )

            return (
                "The region with the highest sales is "
                f"{highest_region}, with total sales of "
                f"{highest_sales:,.0f}."
            )

        # ============================================================
        # HIGHEST SALES PRODUCT
        # ============================================================

        if (
            "product" in q
            and (
                "highest sales" in q
                or "most sales" in q
                or "best sales" in q
            )
        ):

            df = self.load_dataframe()

            product_sales = (
                df.groupby("Product")["Sales"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            highest_product = (
                product_sales.index[0]
            )

            highest_sales = (
                product_sales.iloc[0]
            )

            return (
                "The product with the highest total sales is "
                f"{highest_product}, with sales of "
                f"{highest_sales:,.0f}."
            )

        # ============================================================
        # PRODUCT CATEGORIES
        # ============================================================

        if (
            "product categor" in q
            or "different categor" in q
            or "categories of products" in q
        ):

            df = self.load_dataframe()

            categories = sorted(
                df["Category"]
                .dropna()
                .unique()
                .tolist()
            )

            category_text = ", ".join(
                categories
            )

            return (
                "The product categories in the dataset are: "
                f"{category_text}."
            )

        # ============================================================
        # NO STRUCTURED QUESTION DETECTED
        # ============================================================

        return None

    # ================================================================
    # ASK
    # ================================================================

    def ask(
        self,
        question: str,
        k: int = 4,
    ) -> str:
        """
        Answer a user question.

        Structured sales questions are answered using Pandas
        for exact calculations.

        Other questions use the RAG pipeline.
        """

        if not question or not question.strip():
            return "Please provide a question."

        # ============================================================
        # STEP 1: EXACT STRUCTURED DATA ANALYSIS
        # ============================================================

        structured_answer = (
            self.answer_data_question(
                question
            )
        )

        if structured_answer is not None:
            return structured_answer

        # ============================================================
        # STEP 2: RAG RETRIEVAL
        # ============================================================

        documents = self.retrieve(
            question,
            k=k,
        )

        if not documents:
            return (
                "I don't have enough information in the "
                "knowledge base to answer that question."
            )

        # ============================================================
        # BUILD CONTEXT
        # ============================================================

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # ============================================================
        # LLM PROMPT
        # ============================================================

        prompt = f"""
You are an AI Data Analyst assistant.

Answer the user's question using the provided context.

Rules:

1. Use the context as the primary source of information.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information,
   clearly say so.
4. Give a clear and concise answer.
5. When appropriate, explain the reasoning.
6. Do not make up numerical values.
7. Do not assume missing prices, quantities, or values.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        # ============================================================
        # GENERATE ANSWER
        # ============================================================

        response = self.llm.invoke(
            prompt
        )

        return response.content


# ====================================================================
# FACTORY FUNCTION
# ====================================================================

def create_rag_engine() -> RAGEngine:
    """Create and return a RAG engine instance."""

    return RAGEngine()