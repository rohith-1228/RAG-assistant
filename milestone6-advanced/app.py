import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import hashlib
import re
from datetime import datetime


# ==================================
# Page Config (must be first)
# ==================================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================
# Gemini Setup
# ==================================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file.")
    st.stop()

genai.configure(api_key=api_key)

llm = genai.GenerativeModel("gemini-2.5-flash")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10
CHUNK_SIZE = 500        # characters per chunk (larger than M5 for better context)
CHUNK_OVERLAP = 100     # overlap between chunks to avoid cutting sentences
VECTOR_STORE_DIR = "vector_store"   # folder to save FAISS index and chunks


# ==================================
# Load Embedding Model Once
# (cached so it never reloads)
# ==================================
@st.cache_resource
def load_model():
    """
    Loads the SentenceTransformer embedding model once and caches it.
    Reusing from Milestone 5 — just cached more explicitly.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ==================================
# Utility: Generate a unique hash
# for a list of PDF filenames.
# Used to detect if the same PDFs
# were already processed before.
# ==================================
def get_pdf_hash(pdf_names: list[str]) -> str:
    """
    Creates a unique fingerprint from the sorted list of PDF filenames.
    If the same PDFs are uploaded again, the hash will match and we
    can reload the saved index instead of recomputing embeddings.
    """
    combined = "_".join(sorted(pdf_names))
    return hashlib.md5(combined.encode()).hexdigest()


# ==================================
# Utility: Save vector store to disk
# ==================================
def save_vector_store(index, chunks_with_metadata, pdf_hash: str):
    """
    Saves the FAISS index and chunk metadata to disk so we don't
    have to recompute embeddings every time the app restarts.

    Files saved:
    - vector_store/<hash>.index  → FAISS binary index
    - vector_store/<hash>.json   → chunks with metadata
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    faiss.write_index(
        index,
        os.path.join(VECTOR_STORE_DIR, f"{pdf_hash}.index")
    )

    with open(os.path.join(VECTOR_STORE_DIR, f"{pdf_hash}.json"), "w") as f:
        json.dump(chunks_with_metadata, f)


# ==================================
# Utility: Load vector store from disk
# ==================================
def load_vector_store(pdf_hash: str):
    """
    Loads a previously saved FAISS index and chunk metadata from disk.
    Returns (index, chunks_with_metadata) or (None, None) if not found.
    """
    index_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_hash}.index")
    chunks_path = os.path.join(VECTOR_STORE_DIR, f"{pdf_hash}.json")

    if os.path.exists(index_path) and os.path.exists(chunks_path):

        index = faiss.read_index(index_path)

        with open(chunks_path, "r") as f:
            chunks_with_metadata = json.load(f)

        return index, chunks_with_metadata

    return None, None


# ==================================
# Smart Chunking
# (Upgraded from Milestone 5)
# ==================================
def smart_chunk_text(text: str, filename: str, page_num: int, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[dict]:
    """
    Smarter chunking compared to Milestone 5's simple character split.

    Improvements:
    - Splits on paragraph boundaries first (double newlines)
    - Falls back to sentence boundaries if paragraph is too large
    - Maintains overlap between chunks to preserve context
    - Stores metadata: filename, page number, chunk number

    Each chunk is a dict:
    {
        "text": "...",
        "filename": "doc.pdf",
        "page": 3,
        "chunk_num": 12
    }
    """
    chunks_with_metadata = []

    # Split on paragraph boundaries (double newlines)
    paragraphs = re.split(r'\n\s*\n', text.strip())

    current_chunk = ""
    chunk_num = 0

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding this paragraph keeps us under chunk_size, add it
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + "\n\n"

        else:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks_with_metadata.append({
                    "text": current_chunk.strip(),
                    "filename": filename,
                    "page": page_num,
                    "chunk_num": chunk_num
                })
                chunk_num += 1

                # Keep overlap: carry last `overlap` characters into next chunk
                current_chunk = current_chunk[-overlap:] + paragraph + "\n\n"

            else:
                # Paragraph itself is larger than chunk_size — split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks_with_metadata.append({
                                "text": current_chunk.strip(),
                                "filename": filename,
                                "page": page_num,
                                "chunk_num": chunk_num
                            })
                            chunk_num += 1
                            current_chunk = current_chunk[-overlap:] + sentence + " "
                        else:
                            current_chunk = sentence + " "

    # Save the last remaining chunk
    if current_chunk.strip():
        chunks_with_metadata.append({
            "text": current_chunk.strip(),
            "filename": filename,
            "page": page_num,
            "chunk_num": chunk_num
        })

    return chunks_with_metadata


# ==================================
# Extract Text + Chunk from PDFs
# (Upgraded from Milestone 5)
# ==================================
def extract_and_chunk_pdfs(uploaded_files: list) -> tuple[list[dict], int]:
    """
    Loops through all uploaded PDFs, extracts text page by page,
    and chunks each page with metadata.

    Returns:
    - all_chunks: list of chunk dicts with metadata
    - total_pages: total pages across all PDFs

    Milestone 5 only handled one PDF and lost page/filename info.
    Milestone 6 tracks every chunk's origin.
    """
    all_chunks = []
    total_pages = 0

    for uploaded_file in uploaded_files:

        try:
            reader = PdfReader(uploaded_file)
            total_pages += len(reader.pages)

            for page_num, page in enumerate(reader.pages, start=1):

                extracted = page.extract_text()

                if not extracted or not extracted.strip():
                    continue  # skip empty/unreadable pages

                page_chunks = smart_chunk_text(
                    text=extracted,
                    filename=uploaded_file.name,
                    page_num=page_num
                )

                all_chunks.extend(page_chunks)

        except Exception as e:
            st.warning(f"⚠️ Could not read {uploaded_file.name}: {e}")

    return all_chunks, total_pages


# ==================================
# Build FAISS Index
# (Same logic as M5, now with metadata)
# ==================================
def build_faiss_index(chunks_with_metadata: list[dict]):
    """
    Encodes all chunk texts into embeddings and builds a FAISS index.
    Same approach as Milestone 5 — just now chunks carry metadata.

    Returns the FAISS index and embedding dimension.
    """
    model = load_model()

    texts = [chunk["text"] for chunk in chunks_with_metadata]

    with st.spinner("🔢 Generating embeddings..."):
        embeddings = model.encode(texts, show_progress_bar=False)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, dimension


# ==================================
# Process PDFs (with persistence)
# (Upgraded from Milestone 5)
# ==================================
def process_pdfs(uploaded_files: list):
    """
    Main processing pipeline. Upgraded from Milestone 5's process_pdf().

    New in Milestone 6:
    - Handles multiple PDFs
    - Checks if index already exists on disk (persistent vector store)
    - Saves index to disk after building
    - Returns rich metadata alongside index and chunks
    """
    pdf_names = [f.name for f in uploaded_files]
    pdf_hash = get_pdf_hash(pdf_names)

    # Try loading from disk first (persistent vector store)
    index, chunks_with_metadata = load_vector_store(pdf_hash)

    if index is not None:
        st.sidebar.success("⚡ Loaded from saved index")
        total_pages = len(set(c["page"] for c in chunks_with_metadata))
        return index, chunks_with_metadata, total_pages

    # Not found on disk — build from scratch
    st.sidebar.info("🔄 Building new index...")

    chunks_with_metadata, total_pages = extract_and_chunk_pdfs(uploaded_files)

    if not chunks_with_metadata:
        st.error("❌ No text could be extracted from the uploaded PDFs.")
        st.stop()

    index, _ = build_faiss_index(chunks_with_metadata)

    # Save to disk for next time
    save_vector_store(index, chunks_with_metadata, pdf_hash)

    return index, chunks_with_metadata, total_pages


# ==================================
# Retrieval
# (Upgraded from Milestone 5)
# ==================================
def retrieve_chunks(question: str, index, chunks_with_metadata: list[dict], k=TOP_K) -> tuple[list[dict], list[float]]:
    """
    Encodes the question and searches the FAISS index for top-k matches.

    Improvements over Milestone 5:
    - Returns full chunk dicts (with metadata) not just indices
    - Deduplicates chunks with identical text
    - Returns distance scores for the debug panel

    Returns:
    - top_chunks: list of unique chunk dicts
    - scores: corresponding distance scores
    """
    model = load_model()

    query_embedding = model.encode([question])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k=k
    )

    seen_texts = set()
    top_chunks = []
    scores = []

    for idx, dist in zip(indices[0], distances[0]):

        chunk = chunks_with_metadata[idx]
        text = chunk["text"]

        # Deduplicate: skip if we've seen this exact text
        if text in seen_texts:
            continue

        seen_texts.add(text)
        top_chunks.append(chunk)
        scores.append(float(dist))

    return top_chunks, scores


# ==================================
# Build Prompt
# (Upgraded from Milestone 5)
# ==================================
def build_prompt(question: str, top_chunks: list[dict], chat_history: list[dict]) -> str:
    """
    Builds the Gemini prompt with retrieved context and conversation history.

    Improvements over Milestone 5:
    - Each context block is labeled with its source (filename + page)
    - Better instructions: prefer definitions, admit uncertainty, no hallucination
    - Chat history is formatted more clearly
    """
    # Build context with source labels
    context_blocks = []

    for i, chunk in enumerate(top_chunks, start=1):
        block = (
            f"[Source {i}: {chunk['filename']} | Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    # Format conversation history
    history_text = ""
    for msg in chat_history[-6:]:   # last 3 exchanges to keep prompt focused
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are an expert PDF assistant. Answer questions strictly based on the provided context.

Conversation History:
{history_text}

Retrieved Context:
{context}

Question: {question}

Instructions:
1. Answer ONLY using the context above. Do not use outside knowledge.
2. If the context contains a definition, provide it clearly and directly.
3. If the answer spans multiple sources, synthesize them coherently.
4. If the context is insufficient, say: "The document does not contain enough information to answer this question."
5. Never guess or hallucinate. Accuracy is more important than completeness.
6. Keep answers clear and well-structured. Use bullet points when listing multiple items.
"""

    return prompt


# ==================================
# Format Source Citations
# ==================================
def format_sources(top_chunks: list[dict]) -> str:
    """
    Builds a deduplicated source citation string from retrieved chunks.
    Displayed below every assistant answer.

    Example output:
    📄 OperatingSystems.pdf — Page 18
    📄 Networks.pdf — Page 5
    """
    seen = set()
    sources = []

    for chunk in top_chunks:
        key = (chunk["filename"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append(f"📄 **{chunk['filename']}** — Page {chunk['page']}")

    return "\n".join(sources)


# ==================================
# Session State Initialization
# ==================================
def init_session_state():
    """
    Initializes all session state variables.
    Keeps state clean and predictable across reruns.
    """
    defaults = {
        "messages": [],
        "index": None,
        "chunks_with_metadata": None,
        "total_pages": 0,
        "total_pdfs": 0,
        "embedding_dim": 0,
        "processed_pdf_names": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================================
# Sidebar
# ==================================
def render_sidebar():
    """
    Renders the sidebar with upload status, model info, and controls.
    Upgraded from Milestone 5's minimal sidebar.
    """
    with st.sidebar:

        st.header("📚 RAG Assistant")
        st.divider()

        # Upload Status
        st.subheader("📂 Upload Status")

        if st.session_state.index is not None:
            st.success(f"✅ {st.session_state.total_pdfs} PDF(s) loaded")
            st.metric("Pages", st.session_state.total_pages)
            st.metric("Chunks", len(st.session_state.chunks_with_metadata))
        else:
            st.info("No PDFs processed yet")

        st.divider()

        # Model Info
        st.subheader("🤖 Model Info")
        st.caption(f"Embedding: `{EMBEDDING_MODEL_NAME}`")
        st.caption(f"LLM: `gemini-2.5-flash`")
        st.caption(f"Top-K Retrieval: `{TOP_K}`")

        if st.session_state.embedding_dim:
            st.caption(f"Embedding Dim: `{st.session_state.embedding_dim}`")

        st.divider()

        # Controls
        st.subheader("⚙️ Controls")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 Rebuild Index", use_container_width=True):
            # Clear all state so PDFs get reprocessed from scratch
            for key in ["index", "chunks_with_metadata", "total_pages",
                        "total_pdfs", "embedding_dim", "processed_pdf_names"]:
                st.session_state[key] = None if key not in ["total_pages", "total_pdfs", "embedding_dim"] else 0
            st.session_state.processed_pdf_names = []
            st.rerun()


# ==================================
# Statistics Dashboard
# ==================================
def render_stats_dashboard():
    """
    Shows a row of metric cards summarizing the loaded knowledge base.
    New in Milestone 6 — gives users a quick overview at a glance.
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("📄 PDFs", st.session_state.total_pdfs)
    col2.metric("📃 Pages", st.session_state.total_pages)
    col3.metric("🧩 Chunks", len(st.session_state.chunks_with_metadata) if st.session_state.chunks_with_metadata else 0)
    col4.metric("📐 Embed Dim", st.session_state.embedding_dim or "—")
    col5.metric("🔍 Top-K", TOP_K)


# ==================================
# Retrieval Debug Panel
# ==================================
def render_debug_panel(top_chunks: list[dict], scores: list[float]):
    """
    Expandable panel showing every retrieved chunk with its metadata.
    New in Milestone 6 — helps visualize the RAG pipeline clearly.
    Useful for interviews: shows you understand what retrieval is doing.
    """
    with st.expander("🔍 Retrieval Debug Panel", expanded=False):

        st.caption(f"Retrieved {len(top_chunks)} unique chunks")

        for i, (chunk, score) in enumerate(zip(top_chunks, scores), start=1):

            st.markdown(f"**Result {i}** — `{chunk['filename']}` | Page `{chunk['page']}` | Chunk `{chunk['chunk_num']}` | Score `{score:.4f}`")
            st.text(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))
            st.divider()


# ==================================
# Main Application
# ==================================
def main():

    init_session_state()
    render_sidebar()

    st.title("📚 RAG Chat Assistant — Milestone 6")
    st.caption("Upload PDFs and ask questions. Answers are grounded in your documents.")

    # -------------------------
    # PDF Upload (multiple)
    # -------------------------
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True   # NEW in Milestone 6
    )

    # -------------------------
    # Process PDFs when uploaded
    # -------------------------
    if uploaded_files:

        current_names = sorted([f.name for f in uploaded_files])

        # Only reprocess if the uploaded files changed
        if current_names != st.session_state.processed_pdf_names:

            with st.spinner("📖 Processing PDFs..."):

                index, chunks_with_metadata, total_pages = process_pdfs(uploaded_files)

            # Store everything in session state
            st.session_state.index = index
            st.session_state.chunks_with_metadata = chunks_with_metadata
            st.session_state.total_pages = total_pages
            st.session_state.total_pdfs = len(uploaded_files)
            st.session_state.processed_pdf_names = current_names

            # Store embedding dimension for display
            model = load_model()
            st.session_state.embedding_dim = model.get_sentence_embedding_dimension()

        # -------------------------
        # Stats Dashboard
        # -------------------------
        render_stats_dashboard()
        st.divider()

    # -------------------------
    # Display Chat History
    # -------------------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show timestamp and sources for assistant messages
            if message["role"] == "assistant":
                st.caption(f"🕐 {message.get('timestamp', '')}")
                if message.get("sources"):
                    with st.expander("📎 Sources"):
                        st.markdown(message["sources"])

    # -------------------------
    # Chat Input
    # -------------------------
    if st.session_state.index is None:
        st.info("👆 Upload one or more PDFs to start chatting.")
        return

    question = st.chat_input("Ask a question about your documents...")

    if not question:
        return

    if not question.strip():
        st.warning("Please enter a valid question.")
        return

    # -------------------------
    # Show User Message
    # -------------------------
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # -------------------------
    # Retrieval
    # -------------------------
    top_chunks, scores = retrieve_chunks(
        question,
        st.session_state.index,
        st.session_state.chunks_with_metadata
    )

    if not top_chunks:
        st.error("No relevant chunks found. Try rephrasing your question.")
        return

    # -------------------------
    # Build Prompt
    # -------------------------
    prompt = build_prompt(
        question,
        top_chunks,
        st.session_state.messages
    )

    # -------------------------
    # Streaming Response (NEW in M6)
    # -------------------------
    with st.chat_message("assistant"):

        response_placeholder = st.empty()
        full_response = ""

        try:
            # stream=True makes Gemini return tokens progressively
            stream = llm.generate_content(prompt, stream=True)

            for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")  # typing cursor

            response_placeholder.markdown(full_response)   # final clean render

        except Exception as e:
            full_response = f"❌ Gemini API error: {e}"
            response_placeholder.markdown(full_response)

        # -------------------------
        # Sources + Timestamp
        # -------------------------
        timestamp = datetime.now().strftime("%H:%M:%S")
        sources_text = format_sources(top_chunks)

        st.caption(f"🕐 {timestamp}")

        with st.expander("📎 Sources"):
            st.markdown(sources_text)

    # -------------------------
    # Save Assistant Message
    # -------------------------
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "timestamp": timestamp,
        "sources": sources_text
    })

    # -------------------------
    # Retrieval Debug Panel
    # -------------------------
    render_debug_panel(top_chunks, scores)


# ==================================
# Entry Point
# ==================================
if __name__ == "__main__":
    main()
