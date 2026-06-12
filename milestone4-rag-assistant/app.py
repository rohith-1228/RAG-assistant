import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os


# -----------------------------
# Gemini Setup
# -----------------------------
load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 RAG PDF Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question"
)

search_button = st.button("🔍 Search")


# -----------------------------
# Chunking Function
# -----------------------------
def chunk_text(text, chunk_size=200, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Load Embedding Model Once
# -----------------------------
@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# -----------------------------
# Process PDF Once
# -----------------------------
@st.cache_resource
def process_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    chunks = chunk_text(text)

    model = load_model()

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return text, chunks, index


# -----------------------------
# Main App
# -----------------------------
if uploaded_file:

    with st.spinner("Processing PDF..."):

        text, chunks, index = process_pdf(
            uploaded_file
        )

    st.success("PDF uploaded successfully!")

    st.write(
        f"Characters extracted: {len(text)}"
    )

    st.write(
        f"Total chunks: {len(chunks)}"
    )

    model = load_model()

    # -----------------------------
    # Search
    # -----------------------------
    if search_button and question:

        with st.spinner("Searching..."):

            query_embedding = model.encode(
                [question]
            )

            distances, indices = index.search(
                np.array(query_embedding).astype(
                    "float32"
                ),
                k=10
            )

            # Build Context
            context = ""

            for idx in indices[0]:

                context += chunks[idx]
                context += "\n\n"

            # Prompt
            prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the provided context.

If a direct definition exists, provide the definition first.

Then provide any additional details found in the document.

Context:
{context}

Question:
{question}

Rules:
1. Use only the provided context.
2. Do not make up information.
3. If the answer is not found, say:
   "I could not find the answer in the document."
"""

            # Generate Answer
            response = llm.generate_content(
                prompt
            )

        # -----------------------------
        # Display Answer
        # -----------------------------
        st.subheader("🤖 Answer")

        st.write(
            response.text
        )

        # -----------------------------
        # Retrieved Context
        # -----------------------------
        st.subheader("📄 Retrieved Context")

        for i, idx in enumerate(indices[0]):

            with st.expander(
                f"Result {i+1} (Chunk #{idx})"
            ):

                st.write(
                    chunks[idx]
                )

                st.write(
                    f"Distance Score: {distances[0][i]:.2f}"
                )

            st.divider()