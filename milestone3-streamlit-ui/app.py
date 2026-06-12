import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np

st.title("📚 RAG PDF Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question"
)

search_button = st.button("🔍 Search")

def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []                                             #empty list to store the chunks

    start = 0                                               #start index

    while start < len(text):                                #starting 0 to end of the pdf
        end = start + chunk_size                            #ending of the chunk i.e(0 to 200)

        chunks.append(text[start:end])                      #appending chunks to the list

        start += chunk_size - overlap                       #overlapping chunks so that its is more understandable and both chunks contain important inforamation

    return chunks

#Process PDF

if uploaded_file:
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    st.success("PDF uploaded successfully!")

    st.write("Characters extracted: ",len(text))

    #create chunks
    chunks = chunk_text(text)

    st.write("Total chunks: ",len(chunks))

    #load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    #search
    if search_button and question:

        with st.spinner("Searching..."):

            query_embedding = model.encode([question])

            distance, indices = index.search(
            np.array(query_embedding).astype("float32"),
            k=5
        )

        st.subheader("Relevant Chunks")

        for i, idx in enumerate(indices[0]):

            with st.expander(
                f"Result {i+1} (Chunk #{idx})"
            ):

                st.write(chunks[idx])
                

                st.write(
                    f"Distance score: {distance[0][i]:.2f}"
                )

            st.divider()

