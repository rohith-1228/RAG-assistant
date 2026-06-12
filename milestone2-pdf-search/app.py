from sentence_transformers import SentenceTransformer       #used for embeddings
from pypdf import PdfReader                                 #used to read pdfs without that pdfs are unreadable to python
import faiss                                                #vector database to store embeddings and perform similarity search
import numpy as np

reader = PdfReader("sample.pdf")                            #creating object to read the pdfs 

text = ""

for page in reader.pages:                                   #loop to read every page in the pdf

    extracted = page.extract_text()                         #extract the text in the pages

    if extracted:                                           #some pages only contains images then it returns null.
        text += extracted + "\n"


print("PDF loaded successfully")
print("Characters extracted: ",len(text))


def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []                                             #empty list to store the chunks

    start = 0                                               #start index

    while start < len(text):                                #starting 0 to end of the pdf
        end = start + chunk_size                            #ending of the chunk i.e(0 to 200)

        chunks.append(text[start:end])                      #appending chunks to the list

        start += chunk_size - overlap                       #overlapping chunks so that its is more understandable and both chunks contain important inforamation

    return chunks

chunks = chunk_text(text)                                   #calling the function to make the extracted text into chunks
print(f"Total Chunks: {len(chunks)}")                       #total no.of chunks

model = SentenceTransformer("all-miniLM-L6-v2")             #pre-trained model which understands semantics
embeddings = model.encode(chunks)                           #converts the chunks into vectors

dimension = embeddings.shape[1]                             #to get the dimension of the chunks
index = faiss.IndexFlatL2(dimension)                        #creates a vector database

index.add(                                  
    np.array(embeddings).astype("float32")                  #convert the embeddings into numpy arrays and stores the chunks inside vector database
)

while True:

    query = input("\n Ask a question (or type exit): ")

    if query.lower() == "exit":
        break

    query_embeddings = model.encode([query])                #covert the query into a vector

    distance, indices = index.search(
        np.array(query_embeddings).astype("float32"),       #searchig the similar vectors as the query in the vector database
        k=5                                                 #k=5 retrieve top 5 similar chunks
    )

    print("\nRelevant Chunks:\n")

    for idx in indices[0]:
        print(chunks[idx])
        print("\n" + "=" * 60)