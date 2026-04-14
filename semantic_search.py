import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")

documents = [
    "Python asyncio allows you to write concurrent code using async and await syntax.",
    "LangChain is a framework for building applications powered by language models.",
    "ChromaDB is a free open-source vector database that runs locally on your machine.",
    "Groq provides the fastest free LLM inference API available today.",
    "Embeddings convert text into numerical vectors that capture semantic meaning.",
    "RAG stands for Retrieval Augmented Generation — it lets LLMs answer from your documents.",
    "GitHub is a platform for hosting and sharing code repositories for free.",
    "VS Code is a free code editor by Microsoft with great Python support.",
]

collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in ChromaDB")

def search(query, n_results=3):
    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        print(f"Result {i+1} (distance: {distance:.4f}):")
        print(f"  {doc}")

search("how do I store vectors locally?")
search("what tool gives free fast AI responses?")
search("explain concurrent programming in Python")