from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

print("Env loaded successfully.")
embeddings
print("OpenAI Embedding model initialized successfully.")   

sentences=[
    "Langchain is a great framework for building LLM applications.",
    "I love working with large language models.",       
    "The weather is nice today.",
    "Artificial Intelligence is the future.",
]

def cosine_similarity(vec1,vec2):
    dot_product=numpy.dot(vec1,vec2)
    norm_vec1=numpy.linalg.norm(vec1)
    norm_vec2=numpy.linalg.norm(vec2)
    return dot_product/(norm_vec1*norm_vec2)

#Intialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings
print("Embedding model initialized successfully.")

final_embedding=embeddings.embed_documents(sentences)

#Calculating Similarity between all pairs

for i in range (len(sentences)):
    for j in range(i+1,len(sentences)):
        sim=cosine_similarity(final_embedding[i],final_embedding[j])
        print(f"Cosine Similarity between '{sentences[i]}' and '{sentences[j]}' is: {sim} \n")
