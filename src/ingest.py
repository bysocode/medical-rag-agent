from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from src.config import QDRANT_URL, COLLECTION_NAME


DATA_DIR = Path("data/raw_test")


def load_pdfs():
    documents = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
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


def main():
    documents = load_pdfs()

    if not documents:
        print("Aucun PDF trouvé dans data/raw_test.")
        return

    chunks = split_documents(documents)
    create_vectorstore(chunks)

    print("Indexation terminée dans Qdrant.")


if __name__ == "__main__":
    main()