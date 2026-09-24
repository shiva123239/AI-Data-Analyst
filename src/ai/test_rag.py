from src.ai.rag_engine import RAGEngine


def main():
    print("\n" + "=" * 60)
    print("RAG DATA TEST")
    print("=" * 60)

    # Connect to the existing RAG knowledge base
    rag = RAGEngine()

    # Start with ONE question only
    questions = [
    "What is the total sales in the dataset?",
    "Which region has the highest sales?",
    "Which product has the highest sales?",
    "What are the different product categories?",
]

    for question in questions:
        print("\n" + "-" * 60)
        print("QUESTION:", question)
        print("-" * 60)

        answer = rag.ask(question)

        print("\nANSWER:")
        print(answer)


if __name__ == "__main__":
    main()