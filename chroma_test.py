import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -------------------------
# 1. Read PDF
# -------------------------

pdf_path = "data/Students-Centric-Policies.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"

# -------------------------
# 2. Clean text
# -------------------------

lines = text.splitlines()

clean_text = "\n".join(
    line.strip()
    for line in lines
    if line.strip()
)

# -------------------------
# 3. Same chunking
# -------------------------

chunk_size = 500
chunk_overlap = 50

chunks = []
start = 0

while start < len(clean_text):
    end = start + chunk_size
    chunk = clean_text[start:end]

    chunks.append(chunk)

    start = end - chunk_overlap

# -------------------------
# 4. Local embeddings
# -------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True,
    show_progress_bar=False,
).tolist()

print(f"ALL {len(embeddings)} EMBEDDINGS CREATED")
print(f"EMBEDDING DIMENSION: {len(embeddings[0])}")


# -------------------------
# 6. Create ChromaDB
# -------------------------

chroma_client = chromadb.PersistentClient(
    path="chroma_rag_db",
    settings=Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI"
    )
)

try:
    chroma_client.delete_collection("college_policies")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="college_policies",
    embedding_function=None,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)

print("COLLECTION CREATED")


# -------------------------
# 7. Store Gemini vectors
# -------------------------

print("BEFORE CHROMA UPSERT")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("CHUNKS STORED SUCCESSFULLY!")
print("Number of documents in ChromaDB:", collection.count())