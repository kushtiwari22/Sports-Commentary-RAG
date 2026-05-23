

import os
from pathlib import Path

import pathway as pw
from langchain_community.vectorstores import PathwayVectorClient
from pathway.udfs import DiskCache
from pathway.xpacks.llm import embedders, parsers, splitters
from pathway.xpacks.llm.vector_store import VectorStoreServer

DATA_PATH = Path("./data")
DATA_PATH.mkdir(exist_ok=True)

PATHWAY_HOST: str = os.getenv("PATHWAY_HOST", "0.0.0.0")
PATHWAY_PORT: int = int(os.getenv("PATHWAY_PORT", "8765"))

_vector_server_thread = None  # keep a reference so it isn't GC'd


def ingest_text_file(text_content: str, name: str) -> Path:
    """Write `text_content` to DATA_PATH as a plain .txt file."""
    path = DATA_PATH / f"scraped_data_{name}.txt"
    path.write_text(text_content, encoding="utf-8")
    return path


def build_vector_store() -> VectorStoreServer:
    """
    Construct a Pathway VectorStoreServer watching DATA_PATH/*.txt.
    Call `server.run_server(..., threaded=True)` on the returned object.
    """
    folder_source = pw.io.fs.read(
        path=str(DATA_PATH / "*.txt"),
        format="binary",
        with_metadata=True,
    )

    parser = parsers.UnstructuredParser()
    text_splitter = splitters.TokenCountSplitter(min_tokens=150, max_tokens=450)
    embedder = embedders.OpenAIEmbedder(cache_strategy=DiskCache())

    return VectorStoreServer(
        folder_source,
        embedder=embedder,
        splitter=text_splitter,
        parser=parser,
    )


def start_vector_store() -> None:
    """Start the Pathway VectorStoreServer in a background thread (idempotent)."""
    global _vector_server_thread
    if _vector_server_thread is not None:
        return  # already running
    server = build_vector_store()
    _vector_server_thread = server.run_server(
        PATHWAY_HOST, PATHWAY_PORT, threaded=True
    )


def get_retriever():
    """Return a LangChain retriever backed by the running Pathway VectorStore."""
    client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)
    return client.as_retriever()