"""
rag_pipeline.py
End-to-end RAG pipeline (single-file) using:
- LangChain (document splitting, LLM wrapper)
- OpenAI embeddings via OpenAIEmbeddings
- Chroma DB (local)
- Sparse BM25 (rank_bm25)
- LLM-based query enhancement & re-ranking
- Final generation with citations
"""

import os
import uuid
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Chroma
from langchain_community.vectorstores import Chroma

# Sparse retrieval
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# For pretty output
import json

# Ensure NLTK data available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------------
# 0. Configuration
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Set OPENAI_API_KEY environment variable before running.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

EMBEDDING_MODEL_NAME = "text-embedding-3-large" # OpenAI embedding model
LLM_MODEL_NAME = "gpt-3.5-turbo"  # replace with your model choice (cost/availability)
CHROMA_PERSIST_DIR = "./chroma_data"

# Retrieval params
DENSE_TOP_K = 10
SPARSE_TOP_K = 15
MERGED_TOP_K = 12
RERANK_TOP_K = 8

# Chunking params
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

#Source File Loading
SOURCE_FILE = "./data/langchain.txt"

try:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        SOURCE_TEXT = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")


# -------------------------
# Data classes
# -------------------------
@dataclass
class ChunkRecord:
    id: str
    text: str
    metadata: Dict[str, Any]
    # dense_score and sparse_score used later if needed
    dense_score: float = 0.0
    sparse_score: float = 0.0

# -------------------------
# 2. Ingest & chunk
# -------------------------
def ingest_and_chunk(text: str, chunk_size:int=CHUNK_SIZE, chunk_overlap:int=CHUNK_OVERLAP) -> List[Document]:
    """
    Load the text and split into Document objects with metadata
    """
    doc = Document(page_content=text, metadata={"source": "user_textfile.txt"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents([doc])
    # assign stable ids and metadata with chunk index for citations
    docs_with_meta = []
    for i, d in enumerate(splits):
        meta = dict(d.metadata or {})
        meta.update({
            "chunk_id": f"chunk_{i}",
            "chunk_index": i,
            "source": meta.get("source", "user_textfile.txt")
        })
        docs_with_meta.append(Document(page_content=d.page_content, metadata=meta))
    print(f"[ingest] created {len(docs_with_meta)} chunks")
    return docs_with_meta

# -------------------------
# 3. Build embeddings & Chroma
# -------------------------
def build_dense_index(docs: List[Document], persist_dir: str = CHROMA_PERSIST_DIR):
    """
    Create or load Chroma vector store for documents.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print("[chroma] persisted vector store at", persist_dir)
    return vectordb

# -------------------------
# 4. Build sparse BM25 index
# -------------------------
def build_bm25_index(docs: List[Document]) -> Tuple[BM25Okapi, List[str]]:
    tokenized_corpus = []
    corpus_texts = []
    for d in docs:
        t = d.page_content
        tokens = word_tokenize(t.lower())
        tokenized_corpus.append(tokens)
        corpus_texts.append(t)
    bm25 = BM25Okapi(tokenized_corpus)
    print("[bm25] built BM25 over", len(corpus_texts), "chunks")
    return bm25, corpus_texts

# -------------------------
# 5. Query enhancement (LLM rewrite/expand)
# -------------------------
def enhance_query_with_llm(query: str, llm: ChatOpenAI) -> str:
    """
    Use the LLM to rewrite / expand the user's query for better retrieval.
    Simple prompt: produce a concise improved search query preserving intent.
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a query rewriting assistant. Rewrite the user's query into a concise, explicit search query that improves retrieval, keeping intent unchanged. Output only the rewritten query."
        ),
        HumanMessagePromptTemplate.from_template("{user_query}")
    ])
    formatted = prompt.format_prompt(user_query=query).to_messages()
    resp = llm.invoke(formatted)
    rewritten = resp.content.strip()
    print("[enhance] rewritten query:", rewritten)
    return rewritten

# -------------------------
# 6. Dense + Sparse retrieval & merge
# -------------------------
def dense_retrieve(vectordb: Chroma, query: str, top_k:int = DENSE_TOP_K) -> List[Tuple[int, str, float]]:
    """
    Use Chroma similarity search (dense embeddings) to fetch top_k chunks.
    Returns list of tuples (chunk_index, text, score) — score is similarity (smaller=better for some stores)
    """
    # Using vectordb.similarity_search_with_score
    docs_and_scores = vectordb.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, score in docs_and_scores:
        idx = doc.metadata.get("chunk_index")
        results.append((idx, doc.page_content, score))
    print(f"[dense] retrieved {len(results)} docs")
    return results

def sparse_retrieve(bm25: BM25Okapi, corpus_texts: List[str], query: str, top_k:int = SPARSE_TOP_K) -> List[Tuple[int, str, float]]:
    """
    Use BM25 to get top_k documents. Returns (chunk_index, text, score)
    """
    q_tokens = word_tokenize(query.lower())
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = [(idx, corpus_texts[idx], float(score)) for idx, score in ranked]
    print(f"[sparse] retrieved {len(results)} docs")
    return results

def merge_dense_sparse(dense_results, sparse_results, merged_k: int = MERGED_TOP_K) -> List[ChunkRecord]:
    """
    Merge dense and sparse results into a combined scored list.
    We normalize scores and combine. Then return top merged_k ChunkRecord objects.
    """
    combined = {}
    # Dense: scores can be distance -> lower better; convert to inverse score
    dense_scores = [s for (_,_,s) in dense_results] or [1.0]
    max_dense = max(dense_scores)
    min_dense = min(dense_scores)
    for idx, text, s in dense_results:
        # convert to a positive similarity where higher is better
        # if s is distance, similarity = 1/(1+s) as simple transform
        sim = 1.0/(1.0 + max(s, 1e-9))
        combined[idx] = ChunkRecord(id=f"chunk_{idx}", text=text, metadata={"chunk_index":idx}, dense_score=sim)

    sparse_scores = [s for (_,_,s) in sparse_results] or [1.0]
    max_sparse = max(sparse_scores)
    min_sparse = min(sparse_scores)
    # normalize sparse scores (higher bm25 = more relevant)
    sp_min = min_sparse
    sp_max = max_sparse if max_sparse != sp_min else sp_min+1.0
    for idx, text, s in sparse_results:
        norm = (s - sp_min) / (sp_max - sp_min + 1e-9)
        if idx in combined:
            combined[idx].sparse_score = norm
        else:
            combined[idx] = ChunkRecord(id=f"chunk_{idx}", text=text, metadata={"chunk_index":idx}, sparse_score=norm)

    # compute combined score as weighted sum
    results = list(combined.values())
    for r in results:
        r_score = 0.6 * getattr(r, "dense_score", 0.0) + 0.4 * getattr(r, "sparse_score", 0.0)
        # store combined in metadata for sorting
        r.metadata["combined_score"] = r_score

    # sort by combined_score desc
    ordered = sorted(results, key=lambda x: x.metadata.get("combined_score", 0.0), reverse=True)
    top = ordered[:merged_k]
    print(f"[merge] top {len(top)} merged docs")
    return top

# -------------------------
# 7. Re-rank using LLM
# -------------------------
def rerank_with_llm(candidates: List[ChunkRecord], query: str, llm: ChatOpenAI, top_k:int = RERANK_TOP_K) -> List[ChunkRecord]:
    """
    Ask the LLM to rank the candidates by relevance. We'll build a prompt containing candidate texts (truncated)
    and ask the LLM to return a JSON array of chunk_ids ordered by relevance. Then we reorder.
    """
    # prepare a safe prompt: include up to N candidates with truncated text
    items = []
    for c in candidates:
        snippet = c.text.strip().replace("\n"," ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        items.append({"id": c.id, "text": snippet, "score": c.metadata.get("combined_score", 0.0)})

    system = (
        "You are an assistant that ranks short text chunks by how useful they are to answer the user's question. "
        "Given a query and numbered chunks, return a JSON array of chunk ids ordered most relevant first. "
        "Output only the JSON array, nothing else."
    )
    # 1. Prepare the content string separately
    candidates_str = ""
    for it in items:
        candidates_str += f"- {it['id']}: {it['text']}\n"

    # 2. Use a placeholder {candidates} in the template instead of f-string injection
    human_template = (
        "Query: {query}\n\n"
        "Chunks:\n"
        "{candidates}" 
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    # 3. Pass the content as a variable
    messages = prompt.format_prompt(query=query, candidates=candidates_str).to_messages()
    # 
    resp = llm.invoke(messages)
    # attempt to parse JSON array from response
    text = resp.content.strip()
    try:
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError("expected list")
        # reorder candidates per arr
        id_to_c = {c.id: c for c in candidates}
        ordered = [id_to_c[i] for i in arr if i in id_to_c]
        # append any missing ones preserving original order
        missing = [c for c in candidates if c.id not in arr]
        final = ordered + missing
        final_top = final[:top_k]
        print(f"[rerank] LLM returned {len(final_top)} ranked candidates")
        return final_top
    except Exception as e:
        print("[rerank] failed to parse LLM output as JSON array:", e)
        # fallback: return top-K by original combined_score
        fallback = sorted(candidates, key=lambda x: x.metadata.get("combined_score",0.0), reverse=True)[:top_k]
        return fallback

# -------------------------
# 8. Final answer generation with citations
# -------------------------
def generate_answer_with_citations(query: str, ranked_chunks: List[ChunkRecord], llm: ChatOpenAI, system_prompt: str) -> Dict[str, Any]:
    """
    Create final answer by injecting top chunks (as context) into LLM with a system prompt that requests citations.
    We'll include each chunk with a citation label like [chunk_3].
    """
    # prepare context text with citation labels
    context_parts = []
    for c in ranked_chunks:
        label = c.id
        snippet = c.text.strip()
        # truncate each snippet to a reasonable length to fit tokens
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        context_parts.append(f"[{label}]\n{snippet}\n")

    context_str = "\n\n".join(context_parts)

    system_template = system_prompt.strip()
    human_template = (
        "User question: {query}\n\n"
        "You have the following retrieved context chunks (each labeled). Use them to answer the question. "
        "When you quote or use information from a chunk, append the chunk label in square brackets as a citation. "
        "If information is not present in the context, say 'I don't know based on the provided documents.'\n\n"
        "Context:\n{context}\n\n"
        "Provide:\n1) A concise answer.\n2) A list of citations used (chunk labels)."
    )
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    messages = prompt.format_prompt(query=query, context=context_str).to_messages()
    resp = llm.invoke(messages)
    answer_text = resp.content.strip()
    # Attempt naive extraction of citations (labels in square brackets)
    import re
    cited = re.findall(r"\[(chunk_\d+)\]", answer_text)
    return {
        "answer": answer_text,
        "citations": list(dict.fromkeys(cited)),  # unique in order
        "used_chunks": [{ "id": c.id, "text": (c.text[:400] + "...") if len(c.text)>400 else c.text } for c in ranked_chunks]
    }

# -------------------------
# 9. Orchestration function
# -------------------------
def run_pipeline_interactive():
    # Step A: ingest & chunk
    docs = ingest_and_chunk(SOURCE_TEXT)

    # Step B: build dense index (Chroma)
    vectordb = build_dense_index(docs, persist_dir=CHROMA_PERSIST_DIR)

    # Step C: build sparse index
    bm25, corpus_texts = build_bm25_index(docs)

    # LLM instances for enhancement, rerank, and final generation
    # (we create separate instances but they could be same)
    llm_enhance = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, max_tokens=300)
    llm_rerank = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, max_tokens=300)
    llm_final = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.0, max_tokens=800)

    # System prompt for final answer
    system_prompt = (
        "You are a helpful, concise assistant. Base your answers strictly on the provided context. "
        "Be factual and provide citation labels (e.g., [chunk_3]) for any claim derived from the context."
    )

    print("\n--- RAG system ready. Enter your question (type 'exit' to quit) ---\n")
    while True:
        q = input("User query: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        # Step 1: query enhancement
        rewritten = enhance_query_with_llm(q, llm_enhance)

        # Step 2: dense + sparse retrieval
        dense_hits = dense_retrieve(vectordb, rewritten, top_k=DENSE_TOP_K)
        sparse_hits = sparse_retrieve(bm25, corpus_texts, rewritten, top_k=SPARSE_TOP_K)

        # Step 3: merge results
        merged = merge_dense_sparse(dense_hits, sparse_hits, merged_k=MERGED_TOP_K)

        # Step 4: rerank using LLM
        reranked = rerank_with_llm(merged, rewritten, llm_rerank, top_k=RERANK_TOP_K)

        # Step 5: final answer generation with citations
        output = generate_answer_with_citations(q, reranked, llm_final, system_prompt)

        print("\n--- ANSWER ---\n")
        print(output["answer"])
        print("\n--- CITATIONS ---")
        print(output["citations"])
        print("\n--- Top chunks (preview) ---")
        for c in output["used_chunks"]:
            print(f"{c['id']}: {c['text'][:300].replace('\\n',' ')}")
        print("\n==============================\n")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    run_pipeline_interactive()
