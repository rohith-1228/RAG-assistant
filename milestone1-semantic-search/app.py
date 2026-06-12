from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample knowledge base
documents = [
    "Machine learning is a subset of artificial intelligence.",         #doc 1
    "Python is a popular programming language.",                        #doc 2
    "Neural networks are inspired by the human brain.",                 #doc 3
    "Deep learning is a branch of machine learning.",                   #doc 4
    "Transformers are used in modern language models.",                 #doc 5
    "Large language models are trained on huge amounts of text.",
    "FAISS is a vector database developed by Meta.",
    "Embeddings convert text into numerical vectors.",
    "RAG stands for Retrieval Augmented Generation.",
    "Streamlit is used to build data applications quickly."
]

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")     #pre-trained model understands all the semantics

print("Creating embeddings...")

embeddings = model.encode(documents)            #converts the documents into vectors

# Create FAISS index
dimension = embeddings.shape[1]                 # shape = [10,328] 10 documents 328 dimensions      shape[1]=328

index = faiss.IndexFlatL2(dimension)            #creates an database to store and search the vectors.   L2 -> Euclidean distance (to find similar vectors)

index.add(
    np.array(embeddings).astype("float32")      #adding all the vectors
)

while True:

    query = input("\nAsk a question (or type exit): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])         #converting the question into vector

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),        #searchig the similar vectors as the query in the vector database
        k=3                                                 #k=3 retrieve top 3 similar documents
    )

    print("\nTop Matches:\n")

    for i, idx in enumerate(indices[0], start=1):           #printing the retrived documents.
        print(f"{i}. {documents[idx]}")