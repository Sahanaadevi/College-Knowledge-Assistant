print("EMBEDDING TEST STARTED")

import os
from pypdf import PdfReader
from google import genai
from google.genai import types

pdf_path = "data/Students-Centric-Policies.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"

print("PDF TEXT EXTRACTED")

# Keep the same cleaning used in pdf_test.py
lines = text.splitlines()

clean_text = "\n".join(
    line.strip()
    for line in lines
    if line.strip()
)

print("TEXT CLEANED")

# Keep the same chunking used in pdf_test.py
chunk_size = 500
chunk_overlap = 50

chunks = []
start = 0

while start < len(clean_text):
    end = start + chunk_size
    chunk = clean_text[start:end]
    chunks.append(chunk)
    start = end - chunk_overlap

print("Number of chunks:", len(chunks))

# Gemini API client
client = genai.Client(api_key=os.environ["Gemini_API_Key"])

print("GEMINI CLIENT CREATED")

embeddings = []

for i, chunk in enumerate(chunks):

    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=chunk,
        config=types.EmbedContentConfig(
            output_dimensionality=768
        )
    )

    vector = result.embeddings[0].values
    embeddings.append(vector)

    print(f"Embedded {i + 1}/{len(chunks)}")

print("ALL 71 EMBEDDINGS CREATED")

print("Embedding dimension:", len(embeddings[0]))

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding:")
print(embeddings[0][:10])