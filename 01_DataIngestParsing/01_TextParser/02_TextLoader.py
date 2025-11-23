#TextLoader-Read single File

from langchain_community.document_loaders import TextLoader

loader=TextLoader("data/text_files/doc1.txt", encoding="utf-8")
documents=loader.load()
# print(type(documents))
# print(documents)

print("loaded", len(documents), "documents")
print("content:", documents[0].page_content)
print("metadata:", documents[0].metadata)

