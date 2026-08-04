import os

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings

load_dotenv()


COLLECTION_NAME = "agentic-rag"
PERSIST_DIRECTORY = "./.chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embeddings = OllamaEmbeddings(model = "nomic-embed-text")


def ingest_docs() -> None:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 250, chunk_overlap = 0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    Chroma.from_documents(
        documents = doc_splits,
        collection_name = COLLECTION_NAME,
        embedding = embeddings,
        persist_directory = PERSIST_DIRECTORY,
    )


if not os.path.exists(PERSIST_DIRECTORY):
    ingest_docs()

retriever = Chroma(
    collection_name = COLLECTION_NAME,
    persist_directory = PERSIST_DIRECTORY,
    embedding_function = embeddings,
).as_retriever()
