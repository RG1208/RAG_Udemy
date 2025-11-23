from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
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

#Recursive Character Text Splitter (RECOMMENDED)
text=documents[0].page_content

print("Token  Text Splitter")

token_splitter=TokenTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len
)
token_char_chunks=token_splitter.split_text(text)
print(f"created {len(token_char_chunks)} character based chunks")
print(f"First Chunk:\n{token_char_chunks[0]}")
print("----------------")
print(f"Second Chunk:\n{token_char_chunks[1]}")