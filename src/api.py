from fastapi import FastAPI, HTTPException

from src.schemas import AskRequest, RagResponse, AskResponse
from src.ingest import ingest_documents
from src.rag import ask_rag
from src.agent import ask_agent


app = FastAPI(
    title="Medical RAG Agent",
    description="API pour interroger des documents médicaux avec RAG, Qdrant, LangChain et DeepSeek.",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Medical RAG Agent API is running.",
    }


@app.post("/ingest")
def ingest():
    result = ingest_documents()

    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])

    return result


@app.post("/ask-rag", response_model=RagResponse)
def ask_with_rag(request: AskRequest):
    result = ask_rag(request.question)
    return result


@app.post("/ask-agent", response_model=AskResponse)
def ask_with_agent(request: AskRequest):
    result = ask_agent(request.question)
    return result