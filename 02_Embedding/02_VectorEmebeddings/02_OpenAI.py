from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

print("Env loaded successfully.")
embeddings
print("OpenAI Embedding model initialized successfully.")   


#single text embeddings
single_text="Langchain is a great framework for building LLM applications."
single_embeddings=embeddings.embed_query(single_text)
print(f"length of single text embeddings: {len(single_embeddings)}")
print(f"Single text embeddings: {single_embeddings[:5]}")
