import os

import chromadb
from chromadb.config import Settings
from google import genai
from sentence_transformers import SentenceTransformer


print("RAG TEST STARTED")


# Use the same local model that created the stored 384-dimensional vectors.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Open the already-populated persistent database. Chroma must not create
# embeddings here because the question is embedded locally below.
chroma_client = chromadb.PersistentClient(
	path="chroma_rag_db",
	settings=Settings(
		chroma_api_impl="chromadb.api.segment.SegmentAPI"
	)
)
collection = chroma_client.get_collection(
	name="college_policies",
	embedding_function=None,
)

question = "What is the attendance policy?"
question_embedding = embedding_model.encode(
	question,
	convert_to_numpy=True,
).tolist()

results = collection.query(
	query_embeddings=[question_embedding],
	n_results=3,
	include=["documents", "distances"],
)

retrieved_chunks = results["documents"][0]

print("\nQuestion:")
print(question)
print("\nRetrieved Chunks:")

for index, chunk in enumerate(retrieved_chunks, start=1):
	print(f"\n--- Chunk {index} ---")
	print(chunk)

print(f"\nNumber of retrieved chunks: {len(retrieved_chunks)}")


# Give Gemini only the chunks returned by the similarity search.
context = "\n\n".join(
	f"Chunk {index}:\n{chunk}"
	for index, chunk in enumerate(retrieved_chunks, start=1)
)

prompt = f"""Answer the question using only the provided context.
If the context does not contain the answer, say that the answer is not
available in the retrieved context. Do not use general knowledge.

Question:
{question}

Retrieved context:
{context}
"""

gemini_client = genai.Client(api_key=os.environ["Gemini_API_Key"])
response = gemini_client.models.generate_content(
	model="gemini-flash-lite-latest",
	contents=prompt,
)

print("\nFINAL ANSWER:")
print(response.text)
