from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy
from langchain_huggingface import HuggingFaceEmbeddings

#Semantic Search - retrieving most similar document for a given query

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

print("Env loaded successfully.")
embeddings
print("OpenAI Embedding model initialized successfully.")   


documents=[
    "The Eiffel Tower is located in Paris.",
    "The Great Wall of China is a historic fortification.",
    "The Pyramids of Giza are ancient monuments in Egypt.",
    "The Statue of Liberty is a symbol of freedom in the USA.",
    "Machu Picchu is an ancient Incan city in Peru."
    "Langchain is a great framework for building LLM applications.",
    "I love working with large language models.",       
    "The weather is nice today.",
    "Artificial Intelligence is the future.",
]
Query="Where is the Eiffel Tower located?"

def cosine_similarity(vec1,vec2):
    dot_product=numpy.dot(vec1,vec2)
    norm_vec1=numpy.linalg.norm(vec1)
    norm_vec2=numpy.linalg.norm(vec2)
    return dot_product/(norm_vec1*norm_vec2)

def semantic_search(query,documents,embeddings,top_k=3):
    query_embedding=embeddings.embed_query(query)
    document_embeddings=embeddings.embed_documents(documents)
    
    similarity_scores=[]
    for idx, doc_embedding in enumerate(document_embeddings):
        sim_score=cosine_similarity(query_embedding,doc_embedding)
        similarity_scores.append((sim_score,documents[idx]))
    
    #sort according to top similarity
    similarity_scores.sort(reverse=True)
    return similarity_scores[:top_k]
   
results=semantic_search(Query,documents,embeddings)
# print(results)
for score, doc in results:
    print(f"Score: {score:.4f} | Document: {doc}\n")