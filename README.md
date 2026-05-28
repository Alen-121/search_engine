# 🔍Docs Search Engine

A hybrid search engine built for searching Automation Anywhere blog content. It combines **BM25 keyword search**, **FAISS semantic search**, and **cross-encoder reranking** into a clean Streamlit UI.

---

## Features

- **Keyword Search (BM25)** — fast, exact term matching using `rank-bm25`
- **Semantic Search (FAISS)** — dense vector similarity using `sentence-transformers` embeddings
- **Hybrid Search** — weighted combination of both approaches with a tunable `alpha` parameter
- **Cross-Encoder Reranking** — results are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2` for improved relevance
- **Streamlit UI** — clean, dark-themed interface with per-result score badges and URL links
- **Deduplication** — only the best-matching chunk per blog URL is surfaced

---

## Project Structure

```
search_engine/
├── app.py              # Streamlit UI — search interface and result rendering
├── search.py           # SearchEngine class — BM25, FAISS, hybrid, and reranking logic
├── nlp.py              # Tokenization helpers and file path constants
├── scraper.py          # Web scraper to collect and index blog content
├── chunking_eda.ipynb  # Notebook exploring chunking strategies
├── eda.ipynb           # Exploratory data analysis on the scraped corpus
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## How It Works

1. **Scrape** — `scraper.py` fetches blog posts and saves them as chunks (JSON).
2. **Index** — `nlp.py` builds a BM25 index (pickle) and a FAISS vector index from sentence embeddings.
3. **Search** — `search.py` loads both indexes and runs keyword, semantic, or hybrid search on a query.
4. **Rerank** — hybrid results are optionally reranked by a cross-encoder model.
5. **Display** — `app.py` renders everything in a Streamlit app with score badges and snippet previews.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Alen-121/search_engine.git
cd search_engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Scrape and index blogs

Run the scraper to fetch content and build the BM25 + FAISS indexes:

```bash
python scraper.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

---

## Search Modes

| Mode | Description |
|---|---|
| **Hybrid** (default) | Weighted combination of BM25 + semantic. Adjust the `α` slider in the sidebar. |
| **Keyword (BM25)** | Pure term-frequency matching. Best for exact or specific queries. |
| **Semantic (FAISS)** | Embedding similarity. Best for conceptual or paraphrased queries. |

The sidebar also lets you control how many results to display (3–10).

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `sentence-transformers` | Embedding model + cross-encoder reranking |
| `faiss-cpu` | Approximate nearest-neighbour vector search |
| `rank-bm25` | BM25 keyword index |
| `beautifulsoup4` + `lxml` | HTML parsing in the scraper |
| `requests` | HTTP fetching |
| `nltk` | Tokenization |
| `matplotlib` | Visualizations in EDA notebooks |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Example Queries

- `What is agentic AI?`
- `automation in fintech`
- `IT service management`
- `AI knowledge management`

---

## Notes

- The app will error on startup if the index files are missing — run `scraper.py` first.
- Memory usage is displayed at the top of the app (via `psutil`).
- The search index is cached with `@st.cache_resource` so it only loads once per session.
