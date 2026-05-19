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


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "page inconnue")

        formatted.append(
            f"[Source {i}] Document: {source}, page: {page}\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


def ask(question: str):
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 5
        }
    )

    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_template(
        """
Tu es un assistant spécialisé dans l'analyse de documents médicaux.

Tu dois répondre uniquement à partir du contexte fourni.
Si la réponse n'est pas présente dans le contexte, dis :
"Je ne trouve pas cette information dans les documents fournis."

Tu ne dois pas inventer d'informations médicales.

Question :
{question}

Contexte :
{context}

Réponse structurée en français avec les sources :
"""
    )

    llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

    chain = prompt | llm

    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return response.content, docs


def main():
    print("RAG médical prêt. Tape 'exit' pour quitter.")

    while True:
        question = input("\nQuestion : ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        answer, docs = ask(question)

        print("\nRéponse :")
        print(answer)

        print("\nSources récupérées :")
        for doc in docs:
            print(
                "-",
                doc.metadata.get("source"),
                "page",
                doc.metadata.get("page"),
            )


if __name__ == "__main__":
    main()