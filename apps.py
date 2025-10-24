from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os, sys, re
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.chatbot.server.app import DataIngestionManager, EnsembleRetrieverManager, build_rag_pipeline

app = Flask(__name__, template_folder="client")
import shutil
CORS(app)

rag_chain = None
retriever = None

PERSIST_DIR = "storage"

def clear_storage_and_reset_globals():
    """Delete the persist dir and reset in-memory retriever/rag_chain."""
    global retriever, rag_chain
    try:
        if os.path.exists(PERSIST_DIR):
            print(f"🧹 Deleting storage directory: {PERSIST_DIR}")
            shutil.rmtree(PERSIST_DIR)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        retriever = None
        rag_chain = None
        print("✅ Storage cleared and retriever + rag_chain reset.")
    except Exception as e:
        print("❌ Failed to clear storage:", e)

def is_valid_url(url):
    pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return pattern.match(url) is not None

@app.route("/")
def index():
    print("GET / -> clearing storage and resetting state (page load).")
    clear_storage_and_reset_globals()
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global retriever
    try:
        source = request.form.get('source', '').strip()
        data_ingestion = DataIngestionManager()


        # ✅ Handle URL or file upload
        if source and is_valid_url(source):
            documents = data_ingestion.load_documents(source)
            source_name = source
        else:
            file = request.files.get('file')
            if not file:
                return jsonify({"status": "error", "message": "No file uploaded or URL provided"})

            file_path = f"temp_{file.filename}"
            file.save(file_path)
            documents = data_ingestion.load_documents(file_path)
            os.remove(file_path)  # 🧹 delete temp file after processing
            source_name = file.filename

        if not documents:
            return jsonify({"status": "error", "message": "No documents found in source"})

        # ✅ Build new retriever from scratch
        retriever_manager = EnsembleRetrieverManager()
        retriever = retriever_manager.build_hybrid_retriever(
            docs=documents,
            embedding_model="all-MiniLM-L6-v2",
            bm25_weight=0.5,
            vector_weight=0.5,
            top_k=3,
            persist_dir="storage/"
        )

        return jsonify({
            "status": "success",
            "message": f"Successfully processed: {source_name}",
            "doc_count": len(documents)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/chat", methods=["POST"])
def chat():
    global rag_chain, retriever
    if not retriever:
        return jsonify({"response": "Please upload a document or enter a URL first!"})

    if not rag_chain:
        rag_chain = build_rag_pipeline(llm_model="llama-3.3-70b-versatile", retriever=retriever)

    user_input = request.json.get("message")
    response = rag_chain.invoke({"question": user_input})
    return jsonify({"response": str(response)})

if __name__ == "__main__":
    print("🚀 RAG Chatbot running at http://127.0.0.1:5000")
    

    app.run(debug=True)