import os
import pandas
from typing import List,Dict,Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
print("setup complete")

doc=Document(
    page_content="This is a sample document.",
    metadata= {
        "source": "sample_source",
        "page":1,
        "author":"Rachit Garg",
        "date_created":"2025-22-11"
        }
)
print("Document created:", doc)