"""
Streamlit UI for the Documentation Search Engine.

"""

import streamlit as st
import time
from search import SearchEngine

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AA Docs Search Engine",
    page_icon="🔍",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #FF6B35, #F72585, #7209B7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-header p {
        color: #888;
        font-size: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s, transform 0.2s;
    }

    .result-card:hover {
        border-color: #F72585;
        transform: translateY(-2px);
    }

    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0ff;
        margin-bottom: 0.3rem;
    }

    .result-title a {
        color: #e0e0ff;
        text-decoration: none;
    }

    .result-title a:hover {
        color: #F72585;
    }

    .result-meta {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .badge-score {
        background: linear-gradient(135deg, #F72585, #7209B7);
        color: white;
    }

    .badge-category {
        background: rgba(114, 9, 183, 0.2);
        color: #B5A3F5;
        border: 1px solid rgba(114, 9, 183, 0.4);
    }

    .badge-method {
        background: rgba(255, 107, 53, 0.2);
        color: #FF6B35;
        border: 1px solid rgba(255, 107, 53, 0.4);
    }

    .result-snippet {
        color: #aab;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .result-link {
        color: #7B68EE;
        font-size: 0.8rem;
        word-break: break-all;
    }

    .stats-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stats-label {
        color: #888;
        font-size: 0.8rem;
    }

    .search-time {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Search Engine (cached) ────────────────────────────────────────────

@st.cache_resource
def load_engine():
    return SearchEngine()


try:
    engine = load_engine()
except FileNotFoundError:
    st.error("Index not found! Run `python scraper.py` first to scrape and index blogs.")
    st.stop()


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Search Settings")

    search_mode = st.radio(
        "Search Mode",
        ["Hybrid", "Keyword (BM25)", "Semantic (FAISS)"],
        index=0,
        help="Hybrid combines keyword + semantic for best results"
    )

    if search_mode == "Hybrid":
        alpha = st.slider(
            "Semantic Weight (α)",
            0.0, 1.0, 0.5, 0.1,
            help="Higher = more semantic, Lower = more keyword"
        )
    else:
        alpha = 0.5

    top_k = st.slider("Results to Show", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 📊 Index Stats")
    stats = engine.get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-number">{stats['total_blogs']}</div>
            <div class="stats-label">Blogs</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stats-box">
            <div class="stats-number">{stats['total_chunks']}</div>
            <div class="stats-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top: 1rem; color: #666; font-size: 0.8rem;">
        <strong>Model:</strong> {stats['embedding_model']}<br>
        <strong>Vectors:</strong> {stats['faiss_vectors']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color: #555; font-size: 0.75rem;">
        <strong>How it works:</strong><br>
        • <strong>Keyword</strong>: BM25 exact term matching<br>
        • <strong>Semantic</strong>: Embedding similarity<br>
        • <strong>Hybrid</strong>: Weighted combination
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔍 AA Docs Search Engine</h1>
    <p>Search Automation Anywhere blogs using Keyword, Semantic, or Hybrid search</p>
</div>
""", unsafe_allow_html=True)

# Search input
query = st.text_input(
    "Search",
    placeholder="e.g. What is agentic AI? / automation in fintech / ITSM",
    label_visibility="collapsed",
)

# ─── Search & Display Results ────────────────────────────────────────────────

if query:
    start_time = time.time()

    if search_mode == "Keyword (BM25)":
        results = engine.keyword_search(query, top_k=top_k)
        # Deduplicate by URL
        seen = set()
        deduped = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                deduped.append(r)
        results = deduped[:top_k]
    elif search_mode == "Semantic (FAISS)":
        results = engine.semantic_search(query, top_k=top_k)
        seen = set()
        deduped = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                deduped.append(r)
        results = deduped[:top_k]
    else:
        results = engine.hybrid_search(query, top_k=top_k, alpha=alpha)

    elapsed = time.time() - start_time

    st.markdown(f'<div class="search-time">Found {len(results)} results in {elapsed:.3f}s · Mode: {search_mode}</div>', unsafe_allow_html=True)

    if not results:
        st.info("No relevant results found. Your query may be outside the scope of the indexed blog content. Try a query related to Automation Anywhere topics.")
    else:
        for i, r in enumerate(results):
            # Truncate snippet
            snippet = r["text"][:300]
            if len(r["text"]) > 300:
                snippet += "..."

            # Build score display
            score_pct = min(r["score"] * 100, 100) if search_mode != "Keyword (BM25)" else min(r["score"] / (max(rr["score"] for rr in results) if results else 1) * 100, 100)

            # Extra score details for hybrid
            extra_scores = ""
            if search_mode == "Hybrid" and "keyword_score" in r:
                extra_scores = f' · KW: {r["keyword_score"]:.2f} · Sem: {r["semantic_score"]:.2f}'

            category_display = r.get("category", "general").replace("-", " ").title()

            st.markdown(f"""
            <div class="result-card">
                <div class="result-title">
                    {i+1}. <a href="{r['url']}" target="_blank">{r['title']}</a>
                </div>
                <div class="result-meta">
                    <span class="badge badge-score">Score: {r['score']:.3f}{extra_scores}</span>
                    <span class="badge badge-category">{category_display}</span>
                    <span class="badge badge-method">{r.get('method', search_mode)}</span>
                </div>
                <div class="result-snippet">{snippet}</div>
                <div class="result-link" style="margin-top: 0.5rem;">
                    🔗 <a href="{r['url']}" target="_blank">{r['url']}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Show example queries when no search
    st.markdown("---")
    st.markdown("#### 💡 Try these queries:")

    examples = [
        "What is agentic AI?",
        "automation in fintech",
        "IT service management",
        "AI knowledge management",
        "on-premise enterprise automation",
    ]

    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state["query"] = example
                st.rerun()
