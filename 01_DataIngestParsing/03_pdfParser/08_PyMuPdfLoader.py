from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader

# Fast and Accurate PDF Loader

print("PyMuPDF Loader")
try:
    pdf_loader=PyMuPDFLoader("data/pdf_files/pdf2.pdf")
    documents=pdf_loader.load()
    # print(documents)
    print(f"loaded {len(documents)} documents from PDF using PyMuPDFLoader")
    print(f"First Document Content:\n{documents[0].page_content[:500]}...")

except Exception as e:
    print(f"Error loading PDF with PyPDFLoader: {e}")