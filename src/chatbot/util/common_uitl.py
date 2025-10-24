import os
import sys
from typing import List, Any
from langchain_core.documents.base import Document
from src.chatbot.exception import CustomException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List

def split_documents(
    docs: List[Document], 
    chunk_size: int, 
    chunk_overlap: int
) -> List[Document]:
    
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(docs)
    except Exception as e:
        raise CustomException(e, sys)
    

def format_docs(docs: List[Document]) -> str:
    
    try:
        return "\n\n".join([doc.page_content for doc in docs])    
    except Exception as e:
        raise CustomException(e, sys)