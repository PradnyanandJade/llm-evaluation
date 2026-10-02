from src.rag.ingestion.splitter import splitter
from src.rag.ingestion.loader import loader
from src.rag.vector_store.chroma import vector_store

def get_documents():
    documents = loader.load()
    chunks = splitter.split_documents(
        documents=documents
    )
    lowercase_chunks = []
    return chunks

def ingest_documents():
    documents = get_documents()
    vector_store.add_documents(
        documents=documents
    )

if __name__ == "__main__":
    ingest_documents()