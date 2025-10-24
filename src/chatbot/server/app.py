from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import os, sys
from pathlib import Path
from typing import List, Union, Any, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    AsyncChromiumLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)

from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_transformers import BeautifulSoupTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from src.chatbot.exception import CustomException
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import pickle
import re

load_dotenv()




class DataIngestionManager:

    SUPPORTED_EXTENSIONS = [".txt", 
                            ".pdf", 
                            ".docx", 
                            ".csv", 
                            ".xlsx"
                            ]

    def __init__(self):
        os.environ["USER_AGENT"] = "Mozilla/5.0 (RAG Chatbot Vishnu; +https://github.com/vishnu"


    def load_documents(self, source: Union[str, Path]) -> List[Document]:    

        try:
            source = str(source).strip()
            documents: List[Document] = []

            # 🕸️ 1. Web URL
            if source.startswith(("http://", "https://")):
                loader = WebBaseLoader(source)
                documents.extend(loader.load())
                return documents

            # 📁 2. Local files or directories
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Path not found: {path}")
            
            if path.is_dir():
                for file in path.rglob("*"):
                    if file.suffix.lower() in DataIngestionManager.SUPPORTED_EXTENSIONS:
                        documents.extend(self.load_documents(file))
                return documents

            ext = path.suffix.lower()

            if ext == ".txt":
                loader = TextLoader(str(path), encoding="utf-8")

            elif ext == ".pdf":
                loader = PyPDFLoader(str(path))

            elif ext == ".docx":
                loader = Docx2txtLoader(str(path))

            elif ext == ".csv":
                loader = CSVLoader(file_path=str(path), encoding="utf-8")

            elif ext == ".xlsx":
                loader = UnstructuredExcelLoader(str(path))

            else:
                raise ValueError(f"Unsupported file type: {ext}")

            documents.extend(loader.load())
            return documents
        except Exception as e:
            raise CustomException
        

class EnsembleRetrieverManager:

    def __init__(self):

        pass

    def build_hybrid_retriever(self,
    docs: List[Document],
    embedding_model: str,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    top_k: int = 3,
    persist_dir: str = "storage/"
    ) -> Optional[EnsembleRetriever]:        
        try:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            faiss_path = persist_dir / "faiss_index"
            bm25_path = persist_dir / "bm25.pkl"


            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model
            )
            if faiss_path.exists():
                print("📂 Loading existing FAISS index...")
                vectorstore = FAISS.load_local(
                    str(faiss_path), embeddings, 
                    allow_dangerous_deserialization=True
                )
            else:
                print("⚙️ Creating new FAISS index...")
                vectorstore = FAISS.from_documents(
                    docs, embedding=embeddings
                )
                vectorstore.save_local(str(faiss_path))

            vector_retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": top_k
                }
            )

        
            if bm25_path.exists():
                print("📂 Loading existing BM25 retriever...")
                with open(bm25_path, "rb") as f:
                    bm25_retriever = pickle.load(f)
            else:
                print("⚙️ Creating new BM25 retriever...")
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = top_k
                with open(bm25_path, "wb") as f:
                    pickle.dump(bm25_retriever, f)

 
            retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, vector_weight]
            )

            return retriever
        
        except Exception as e:
            raise CustomException(e, sys)
        
    

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag_pipeline(llm_model: str, retriever) -> Optional[RunnableSequence]:

    try:
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant. Use the following context
        to answer the question. If you don't know say 
        "pandi karumpara pulayadi mone !!!".

        Context:
        {context}

        Question:
        {question}
        """)

        llm = ChatGroq(model=llm_model, temperature=0.7)

        
        rag_chain = (
           {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),  
                "question": lambda x: x["question"]
            }
            | prompt
            | llm
            | StrOutputParser()
        )        

        return rag_chain

    except Exception as e:
        raise CustomException(e, sys)