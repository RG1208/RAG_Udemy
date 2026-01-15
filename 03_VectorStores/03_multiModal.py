import fitz  
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

#Embed Functions

def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data,str): #if path
        image = Image.open(image_data).convert("RGB")
    else: #if PIL image
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features= clip_model.get_image_features(**inputs)
        #normalize embeddings into unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    """Embed Text Using Clip"""
    inputs= clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77 #clip's max token length
    )
    with torch.no_grad():
        features= clip_model.get_text_features(**inputs)
        #normalize embeddings into unit vector
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# Load Data
pdf_path="./data/pdf1.pdf"
doc=fitz.open(pdf_path)
#storage for all documents and embeddings
all_docs=[]
all_embeddings=[]
image_data_store={} #Store actual image data for llm


#Text Splitter
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
print(f"{doc}")

for i, page in enumerate(doc):

    #process text
    text=page.get_text()
    if text.strip():
        #create temporary docuemnt for splitting
        temp_doc= Document(page_content=text, metadata={"page":i, "type":"text"})
        text_chunks= splitter.split_documents([temp_doc])

        #Embedd each chunk using CLIP
        for chunk in text_chunks:
            embedding=embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)

    #Process Image
    # follow these three imp steps

    # Convert pdf image to pil format
    # store as base64 for gpt 4v
    # create clip embedding for retrieval

for img_index, img in enumerate(page.get_images(full=True)):
    try:
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Create unique identifier
        image_id = f"page_{i}_img_{img_index}"

        # Store image as base64 for later use with GPT-4V
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        image_data_store[image_id] = img_base64

        # Embed image using CLIP
        embedding = embed_image(pil_image)
        all_embeddings.append(embedding)

        # Create document for image
        image_doc = Document(
            page_content=f"[Image: {image_id}]",
            metadata={"page": i, "type": "image", "image_id": image_id}
        )
        all_docs.append(image_doc)

    except Exception as e:
        print(f"Error processing image {img_index} on page {i}: {e}")
        continue

doc.close()

print(f"{all_docs}")

# Creating vector store

#Create unified FAISS Vector store with clip embeddings
embeddings_array= np.array(all_embeddings)

#  create custom FAISS index since we have precomputed embeddings
vector_store=FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc,emb in zip(all_docs,embeddings_array)],
    embedding=None, #We are using pre computed embeddings
    metadatas=[doc.metadata for doc in all_docs]
)

#Initialize GPT-4 vision llm model
llm = init_chat_model("openai:gpt-4.1")

def retrieve_multimodal(query,k=5):
    """unified retirveal usign CLIP embeddings for both text and images"""
    #Embed query using CLIP
    query_embeddings=embed_text(query)

    #search based on the query embeddings
    results= vector_store.similarity_search_by_vector(
        embedding=query_embeddings,
        k=k
    )

    return results