from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_community.document_loaders import DirectoryLoader 

loader = DirectoryLoader(
    path="./src/rag/ingestion/data",
    # glob="*.pdf",
    glob="*.txt",
    loader_cls=TextLoader
)

