from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import QDRANT_URL, COLLECTION_NAME


DATA_DIR = Path("data/raw_test")


def load_pdfs(data_dir: Path = DATA_DIR):
    documents = []

    for pdf_path in data_dir.glob("*.pdf"):
        print(f"Chargement du PDF : {pdf_path}")

        loader = PyPDFLoader(str(pdf_path))
        pdf_documents = loader.load()

        for doc in pdf_documents:
            doc.metadata["source"] = pdf_path.name

        documents.extend(pdf_documents)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"Nombre de chunks créés : {len(chunks)}")

    return chunks


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )

    return vectorstore


def ingest_documents():
    documents = load_pdfs()

    if not documents:
        return {
            "status": "error",
            "message": "Aucun PDF trouvé dans data/raw_test.",
            "chunks": 0,
        }

    chunks = split_documents(documents)
    create_vectorstore(chunks)

    return {
        "status": "success",
        "message": "Indexation terminée dans Qdrant.",
        "chunks": len(chunks),
    }


def main():
    result = ingest_documents()
    print(result)


if __name__ == "__main__":
    main()