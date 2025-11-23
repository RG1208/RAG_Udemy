from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from typing import List

class SmartPreProcessor:
    """Advanced PDF preprocessor with error handling"""
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[" "],
            length_function=len
        )
    
    def process_pdf(self, file_path:str)->list[Document]:
        """Process a PDF file and return text chunks"""
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Process each page and split into chunks
        processed_chunks = []

        for page_num,page in enumerate(pages):
        #clean text
            cleaned_text = self._clean_text(page.page_content)
            
            #skip nearly empty pages
            if len(cleaned_text.strip()) < 50:
                continue

            # create chunks with enhanced meta data
            chunks=self.text_splitter.create_documents(
            texts=[cleaned_text],
            metadatas=[{
                **page.metadata,
                "page": page_num + 1,
                "total_pages": len(pages),
                "chunk_method": "SmartPreProcessor",
                "char_count": len(cleaned_text)
                }]
            )
            processed_chunks.extend(chunks)

        return processed_chunks

    def _clean_text(self, text:str)->str:
        """Clean text by removing extra whitespace and unwanted characters"""
        # Remove multiple spaces and newlines
        text = " ".join(text.split())
        text=text.replace("ﬁ","fi")
        text=text.replace("ﬂ","fl")

        return text
        
preprocessor=SmartPreProcessor()

# process a sample PDF if available

try:
    smart_chunks=preprocessor.process_pdf("data/pdf_files/pdf2.pdf")
    print(f"Processed Chunks:{len(smart_chunks)}")

    if smart_chunks:
        print("smart chunks meta data example:")
        for key, value in smart_chunks[0].metadata.items():
            print(f"{key}: {value}")

except Exception as e:
    print(f"Error processing PDF: {e}")
