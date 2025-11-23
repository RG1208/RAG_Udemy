from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader

#Most Common PDF Loader
print("PyPDF Loader")
try:
    pdf_loader=PyPDFLoader("data/pdf_files/pdf2.pdf")
    documents=pdf_loader.load()
    # print(documents)
    print(f"loaded {len(documents)} documents from PDF using PyPDFLoader")
    print(f"First Document Content:\n{documents[0].page_content[:500]}...")

except Exception as e:
    print(f"Error loading PDF with PyPDFLoader: {e}")