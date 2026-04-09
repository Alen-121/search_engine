"""
NLP Processing + Indexing for Blog Search Engine.

Handles chunking, tokenization, and building BM25/FAISS indices.
"""

import os
import json
import pickle
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords', quiet=True)

''' Configuration '''

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
BM25_FILE = os.path.join(DATA_DIR, "bm25_index.pkl")
FAISS_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100
MAX_PARA = 3


''' Chunking '''


'''
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation near the end
            for sep in ['. ', '.\n', '? ', '!\n', '\n\n']:
                idx = text.rfind(sep, start + chunk_size // 2, end + 50)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 50:  # Skip very short chunks
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks
'''

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=MAX_PARA):
    """
    Paragraph-based chunking.
    Groups 2-3 paragraphs per chunk, respecting natural boundaries.
    Falls back to sentence-boundary splitting for very long paragraphs.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 30]

    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        # If adding this paragraph exceeds limits, save current chunk
        if current_chunk and (len(current_chunk) >= overlap or current_len + len(para) > chunk_size):
            chunks.append('\n'.join(current_chunk))
            # Keep last paragraph as overlap (paragraph-level overlap)
            current_chunk = current_chunk[-overlap:]
            current_len = sum(len(p) for p in current_chunk)

        current_chunk.append(para)
        current_len += len(para)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


''' Tokenization '''


STOP_WORDS = set(stopwords.words('english'))


def tokenize(text):
    """Tokenizer for BM25 with NLTK stop word removal."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    # Remove very short tokens and stop words
    return [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]


''' Processing + Indexing '''

def process_blogs(blogs):
    """Process blogs into searchable chunks."""
    print(f"Chunking {len(blogs)} blogs...")
    all_chunks = []

    for blog in blogs:
        text_chunks = chunk_text(blog["content"])

        for i, chunk_text_content in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{len(all_chunks)}",
                "title": blog["title"],
                "url": blog["url"],
                "category": blog["category"],
                "text": chunk_text_content,
                "chunk_index": i,
            })

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(all_chunks)} chunks → {CHUNKS_FILE}")
    return all_chunks


def build_indices(chunks):
    """Build BM25 and FAISS indices."""

    # ── BM25 Index ──
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    # ── FAISS Index ──
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim since normalized)
    index.add(embeddings)

    faiss.write_index(index, FAISS_FILE)

    return bm25, index
