from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from src.rag import get_retriever, format_docs


@tool
def search_medical_documents(question: str) -> str:
    """
    Recherche les passages médicaux les plus pertinents dans la base documentaire.
    Utilise cet outil pour toute question nécessitant des informations issues des PDF.
    """
    retriever = get_retriever(k=5)
    docs = retriever.invoke(question)

    if not docs:
        return "Aucun passage pertinent trouvé dans les documents."

    return format_docs(docs)


@tool
def summarize_retrieved_context(context: str) -> str:
    """
    Résume un contexte médical récupéré depuis les documents.
    Utilise cet outil quand l'utilisateur demande un résumé ou une synthèse.
    """
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )

    messages = [
        (
            "system",
            "Tu es un assistant qui résume des documents médicaux de manière claire et factuelle.",
        ),
        (
            "human",
            f"Résume le contexte suivant en français, sans inventer d'information :\n\n{context}",
        ),
    ]

    response = llm.invoke(messages)
    return response.content


@tool
def verify_answer_with_sources(answer_and_sources: str) -> str:
    """
    Vérifie si une réponse est cohérente avec les sources fournies.
    Utilise cet outil avant de donner une réponse finale sur un sujet médical sensible.
    """
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )

    messages = [
        (
            "system",
            """
Tu es un vérificateur de réponses médicales.
Tu dois dire si la réponse est supportée par les sources.
Réponds avec :
- SUPPORTED si la réponse est bien appuyée par les sources
- PARTIAL si seulement une partie est appuyée
- UNSUPPORTED si la réponse invente des éléments
Puis explique brièvement.
""",
        ),
        (
            "human",
            answer_and_sources,
        ),
    ]

    response = llm.invoke(messages)
    return response.content


def build_agent():
    llm = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=2,
    )

    tools = [
        search_medical_documents,
        summarize_retrieved_context,
        verify_answer_with_sources,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
Tu es un agent d'analyse documentaire médicale.

Tu peux utiliser des outils pour rechercher dans des documents médicaux,
résumer des passages et vérifier tes réponses.

Règles strictes :
1. Pour toute question médicale, utilise d'abord search_medical_documents.
2. Ne réponds jamais uniquement avec tes connaissances générales.
3. Si les sources ne contiennent pas l'information, dis-le clairement.
4. Cite toujours les documents et pages utilisés si disponibles.
5. Ne donne pas de diagnostic médical ni de recommandation thérapeutique personnelle.
6. Pour une question complexe, vérifie ta réponse avec verify_answer_with_sources avant de finaliser.
"""
    )

    return agent


def ask_agent(question: str) -> dict:
    agent = build_agent()

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        }
    )

    answer = result["messages"][-1].content

    return {
        "question": question,
        "answer": answer,
    }


def main():
    print("Agent médical prêt. Tape 'exit' pour quitter.")

    while True:
        question = input("\nQuestion : ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        result = ask_agent(question)
        print("\nRéponse agent :")
        print(result["answer"])


if __name__ == "__main__":
    main()