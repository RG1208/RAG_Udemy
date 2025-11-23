from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredCSVLoader

# Method 1: Using CSVLoader (Row Based)
print("Method 1: Using CSVLoader (Row Based)")

csv_loader = CSVLoader(
    file_path="../data/structured_files/products.csv", 
    encoding="utf-8",
    csv_args={"delimiter": ",", "quotechar": '"'}
)

csv_docs= csv_loader.load()
print("-------------------------")
print(csv_docs)
print("-------------------------")
print(f"loaded {len(csv_docs)} documents using CSVLoader")
print("-------------------------")
for i, doc in enumerate(csv_docs):
    print(f"Document {i+1} content:\n{doc.page_content}\n")
    print(f"Document {i+1} metadata:\n{doc.metadata}\n")
    print("-------------------------")