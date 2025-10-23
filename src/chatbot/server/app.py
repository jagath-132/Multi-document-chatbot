from __future__ import annotations
import os, sys
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    WebBaseLoader
)
from langchain_core.documents.base import Document
from src.chatbot.exception import CustomException
