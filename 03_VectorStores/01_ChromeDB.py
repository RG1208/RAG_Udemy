import os
from dotenv import load_dotenv
import shutil

load_dotenv()

#import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

#vector store
from langchain_community.vectorstores import Chroma

#utility imports
import numpy as np
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# import langchain (optional for later steps)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# vector store
from langchain_community.vectorstores import Chroma

# utility imports
import numpy as np
from typing import List

# ----------------------------------------------------------------------------------------------------------------------

# 1. Creating sample data(documents)
sample_docs = [
    """
    Machine Learning Fundamentals
    Machine Learning is a subset of artificial intelligence that focuses on building systems capable of learning from data without being explicitly programmed for every rule. Instead of following static instructions, machine learning algorithms identify patterns and relationships within large datasets to make predictions or decisions. Common techniques include supervised learning, where models train on labeled data to map inputs to outputs, and unsupervised learning, which finds hidden structures in unlabeled data. This technology powers diverse applications, from email spam filters and recommendation engines to fraud detection systems.
    """,

    """
    Deep Learning/Neural Network Fundamentals
    Deep Learning is a specialized branch of machine learning inspired by the structure and function of the human brain, known as artificial neural networks. These networks consist of layers of interconnected nodes (neurons) that process information hierarchically. While traditional machine learning may struggle with unstructured data like images or audio, deep learning excels at automatically extracting features through its multi-layered architecture. This ability to learn complex, non-linear representations makes deep learning the foundation for modern breakthroughs in computer vision, autonomous driving, and advanced speech synthesis.
    """,

    """
    Natural Language Processing (NLP) Fundamentals
    Natural Language Processing (NLP) is the domain of AI concerned with the interaction between computers and human language. It combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to understand, interpret, and generate human text and speech. NLP tasks range from basic operations like tokenization and part-of-speech tagging to complex challenges such as sentiment analysis, machine translation, and question-answering. By bridging the gap between human communication and computer understanding, NLP enables technologies like virtual assistants, chatbots, and real-time language translators.
    """,
]


# write sample docs to files named data/doc_0.txt, data/doc_1.txt, ...
os.makedirs("data", exist_ok=True)

for idx, content in enumerate(sample_docs):
    file_path = os.path.join("data", f"doc{idx}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
print("Sample Files created")


# ----------------------------------------------------------------------------------------------------------------------

# 2. Document Loading
from langchain_community.document_loaders import DirectoryLoader

loader=DirectoryLoader(
    "data",
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding":"utf-8"}
    )
documents=loader.load()

print(f"{len(documents)} documents loaded successfully.")
# print(f"Sample Document Content:\n{documents[0].page_content[:200]}...\n")
# print(f"Sample Document Content:\n{documents[1].page_content[:200]}...\n")
# print(f"Sample Document Content:\n{documents[2].page_content[:200]}...\n")

# ----------------------------------------------------------------------------------------------------------------------

#3. Document Splitting
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    separators=[" ","\n","\n\n",". " ]
)
chunks=text_splitter.split_documents(documents)
# print(f"{len(chunks)} chunks created from {len(documents)} documents successfully.")
# print(f"Sample Chunk Content:\n{chunks[0].page_content}...\n")

# ----------------------------------------------------------------------------------------------------------------------
#4. Creating Embeddings

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

embeddings=OpenAIEmbeddings()

# ----------------------------------------------------------------------------------------------------------------------
# 5. Creating Vector Store using ChromaDb and storing chunks in vector representation
persist_directory="./chromadb_data"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)  # This deletes the old DB every time you run
    print(f"Deleted existing directory: {persist_directory}")

vectordb=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name="rag_collection"
)

print(f"Vector Store created with {vectordb._collection.count()} vectors and persisted at {persist_directory} successfully.")

# ----------------------------------------------------------------------------------------------------------------------
# Test Similarity Search
query="what is nlp and machine learning"
similar_docs=vectordb.similarity_search(query,k=3)
print("Top 2 similar documents for the query: ")
# for idx, doc in enumerate(similar_docs):
    # print(f"Document {idx+1} Content:\n{doc.page_content}\n")

# ----------------------------------------------------------------------------------------------------------------------
# Advanced search with scores
# results_score = vectordb.similarity_search_with_score(query, k=3)
print("Top 3 similar documents with scores for the query: ")
# for idx, (doc, score) in enumerate(results_score):
    # print(f"Document {idx+1} | Score: {score} | Content:\n{doc.page_content}\n")
# ----------------------------------------------------------------------------------------------------------------------