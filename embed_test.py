from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Python is great for AI development",
    "I love eating pizza on weekends",
    "Machine learning uses data to make predictions",
    "My cat sleeps all day long"
]

embeddings = model.encode(sentences)

print(f"Each embedding has {len(embeddings[0])} dimensions")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")