from pypdf import PdfReader

pdf_path = "data/Students-Centric-Policies.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text() or ""
    text += page_text + "\n"

# Clean unnecessary blank lines
lines = text.splitlines()

clean_text = "\n".join(
    line.strip()
    for line in lines
    if line.strip()
)

# Chunking
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

print("\nFirst chunk:\n")
print(chunks[0])