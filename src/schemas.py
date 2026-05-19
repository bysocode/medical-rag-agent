from pydantic import BaseModel


class IngestRequest(BaseModel):
    file_path: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str


class SourceResponse(BaseModel):
    source: str
    page: int | str
    content_preview: str


class RagResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceResponse]