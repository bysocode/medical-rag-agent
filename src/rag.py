from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from src.config import QDRANT_URL, COLLECTION_NAME


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def get_retriever(k: int = 5):
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
        }
    )


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "page inconnue")

        formatted.append(
            f"[Source {i}] Document: {source}, page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def format_sources(docs):
    sources = []

    for doc in docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "source inconnue"),
                "page": doc.metadata.get("page", "page inconnue"),
                "content_preview": doc.page_content[:300],
            }
        )

    return sources


def ask_rag(question: str):
    retriever = get_retriever(k=5)
    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
Tu es un assistant spécialisé dans l'analyse de documents médicaux.

Tu dois répondre uniquement à partir du contexte fourni.
Si la réponse n'est pas présente dans le contexte, dis :
"Je ne trouve pas cette information dans les documents fournis."

Tu ne dois pas inventer d'informations médicales.
Tu dois citer les sources utilisées.

Question :
{question}

Contexte :
{context}

Réponse structurée en français :
"""
    )

    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "question": question,
        "answer": response.content,
        "sources": format_sources(docs),
    }


def main():
    print("RAG médical DeepSeek prêt. Tape 'exit' pour quitter.")

    while True:
        question = input("\nQuestion : ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        result = ask_rag(question)

        print("\nRéponse :")
        print(result["answer"])

        print("\nSources :")
        for source in result["sources"]:
            print("-", source["source"], "page", source["page"])


if __name__ == "__main__":
    main()