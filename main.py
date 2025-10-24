from src.chatbot.server.app import DataIngestionManager, EnsembleRetrieverManager, build_rag_pipeline
from src.chatbot.util.common_uitl import split_documents
def main():
    # with open(r"C:\Users\VISHNU\Desktop\rag_chat_bot\config\config.yaml", "r") as f:
    #     cfg = yaml.safe_load(f)

    print("🚀 Loading data...")
    docs = DataIngestionManager().load_documents(source=r"C:\Users\JAGATH J G\Desktop\document chatbot\document-chatbot\data")

    print("✂ Splitting documents...")
    chunks = split_documents(docs=docs,
                             chunk_overlap=500,
                             chunk_size=1000)

    print("🔍 Building hybrid retriever (with persistence)...")
    retrievers = EnsembleRetrieverManager()
    retriever = retrievers.build_hybrid_retriever(
        docs= chunks,
        embedding_model = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight=0.5,
        vector_weight=0.5,
        top_k = 3,
        persist_dir = "storage/"
        )
    print("🧠 Creating RAG pipeline...")
    rag = build_rag_pipeline(llm_model="llama-3.3-70b-versatile", retriever= retriever)

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag.invoke({"question": query})
        print("\n💬 Answer:\n", answer)


if __name__ == "_main_":
    main()