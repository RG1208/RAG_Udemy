from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
from typing import List
from langchain_core.documents import Document
import pandas as pd 

#Method 2: Using Custom CSV Processor for better control
print("Method 2: Using Custom CSV Processor")

def process_csv(file_path: str, encoding: str = "utf-8", csv_args: dict = None) -> List[Document]:
    df=pd.read_csv(file_path)
    documents = []
    
    # strategy: create one document per row
    for index, row in df.iterrows():
        content = f"""
        Name: {row['product']}
        Price: {row['price']}
        Description: {row['description']}
        Category: {row['category']}
        Stock: {row['stock']}
        """

        #create document with rich metadata 
        doc= Document(
            page_content=content,
            metadata={
                "source": file_path,
                "row_index": index,
                "name": row['product'],
                "category": row['category'],
                "price": row['price'],
                "data_type": "product_info",
                "stock": row['stock'],
                "quantity": row['quantity']
            }
        )
        documents.append(doc)
    return documents

process_csv("../data/structured_files/products.csv")
print(process_csv("../data/structured_files/products.csv"))                          # prints all Document objects
# print(len(process_csv("../data/structured_files/products.csv")))                     # should print: 3
# print(process_csv("../data/structured_files/products.csv")[0].page_content)          # print first document content