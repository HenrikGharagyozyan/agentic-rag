import time

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_ollama import OllamaEmbeddings

load_dotenv()


COLLECTION_NAME = "agentic-rag"
PERSIST_DIRECTORY = "./.chroma"

# The Gemini free tier allows 100 embedding requests per minute, so documents are
# embedded in batches with a pause between them.
EMBED_BATCH_SIZE = 100
EMBED_BATCH_PAUSE = 60

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# embeddings = OllamaEmbeddings(model = "nomic-embed-text")
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


vectorstore = Chroma(
    collection_name = COLLECTION_NAME,
    persist_directory = PERSIST_DIRECTORY,
    embedding_function = embeddings,
)


def ingest_docs() -> None:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 250, chunk_overlap = 0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    for start in range(0, len(doc_splits), EMBED_BATCH_SIZE):
        vectorstore.add_documents(doc_splits[start : start + EMBED_BATCH_SIZE])
        if start + EMBED_BATCH_SIZE < len(doc_splits):
            time.sleep(EMBED_BATCH_PAUSE)


# Checking the collection rather than the directory: a failed run leaves an empty
# store behind, and that must not be mistaken for a finished ingest.
if not vectorstore.get(limit = 1)["ids"]:
    ingest_docs()

retriever = vectorstore.as_retriever()
