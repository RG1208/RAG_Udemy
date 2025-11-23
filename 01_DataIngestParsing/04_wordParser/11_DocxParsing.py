from docx import Document as DocxDocument
import os
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

print("Method 1: Using Docx2txtLoader")

try:
    docx_loader = Docx2txtLoader("data/word_files/docx1.docx")
    docs=docx_loader.load()
    print(f"Number of documents loaded: {len(docs)}")
    print(f"Content of the first document:\n{docs[0].page_content[:100]}...")
    print(f"Metadata of the first document:\n{docs[0].metadata}")

except Exception as e:
    print(f"Error loading document with Docx2txtLoader: {e}")