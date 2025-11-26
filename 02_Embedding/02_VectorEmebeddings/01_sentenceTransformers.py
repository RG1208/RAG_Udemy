# creating first embeddings

from langchain_huggingface import HuggingFaceEmbeddings

#Intialize embedding model

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings
print("Embedding model initialized successfully.")

# creating first embedding 
text="Hello i am learning about sentence transformers"

final_embedding=embeddings.embed_query(text)
print(f"Text:{text}")
print(f"Embedding:{final_embedding}")
print(f"Embedding Dimension:{len(final_embedding)}")
