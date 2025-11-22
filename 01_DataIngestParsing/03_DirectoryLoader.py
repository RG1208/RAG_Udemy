##Directory Loader- Read multiple files
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

dir_loader=DirectoryLoader(
    "data/text_files/", 
    glob="*.txt", 
    loader_cls=TextLoader, 
    loader_kwargs={"encoding":"utf-8"},
    show_progress=True
    )

documents=dir_loader.load()
print("loaded", len(documents), "documents from directory")
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("len:", len(doc.page_content))