# RAG (Retrieval-Augmented Generation) Architecture:

1. **Document Loading:** Load documents from various sources
2. **Document Splitting:** Break documents into smaller chunks
3. **Embedding Generation:** Convert chunks into vector representations
4. **Vector Storage:** Store embeddings in ChromaDB
5. **Query Processing:** Convert user query to embedding
6. **Similarity Search:** Find relevant chunks from vector store
7. **Context Augmentation:** Combine retrieved chunks with query
8. **Response Generation:** LLM generates answer using context

## Benefits of RAG:

- Reduces hallucinations
- Provides up-to-date information
- Allows citing sources
- Works with domain-specific knowledge