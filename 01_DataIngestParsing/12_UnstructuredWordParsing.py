from docx import Document as DocxDocument
import os
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader

print ("Method 2: Using UnstructuredWordDocumentLoader")

try:
    unstructured_loader = UnstructuredWordDocumentLoader("data/word_files/docx1.docx",mode="elements")
    unstructured_docs = unstructured_loader.load()

    print(f"loaded {len(unstructured_docs)} documents using UnstructuredWordDocumentLoader")

    for i, doc in enumerate(unstructured_docs):
        print(f"Document {i+1} content:\n{doc.page_content}\n")
        print(f"Document {i+1} metadata:\n{doc.metadata}\n")
        print(f"content: {doc.page_content[:100]}")


except Exception as e:
    print(f"An error occurred: {e}")