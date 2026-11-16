from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from pydantic import BaseModel, Field

from datheon.rag import User, citations_for, grounded_answer, retrieve
from datheon.vector_store import ChromaVectorStore, LocalSentenceEmbedder

load_dotenv()


def _configured_users() -> dict[str, User]:
    raw = os.getenv("DATHEON_API_KEYS", "{}")
    try:
        values = json.loads(raw)
        return {
            token: User(
                department=profile["department"],
                allowed_tiers=tuple(profile["allowed_tiers"]),
            )
            for token, profile in values.items()
        }
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("DATHEON_API_KEYS must map tokens to department and allowed_tiers") from error


api_keys = _configured_users()
bearer = HTTPBearer(auto_error=False)
app = FastAPI(title="Datheon Access-Controlled RAG", version="1.0.0")
store: ChromaVectorStore | None = None
openai_client = OpenAI() if os.getenv("OPENAI_API_KEY") else None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


def current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required")
    user = api_keys.get(credentials.credentials)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return user


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest, user: Annotated[User, Depends(current_user)]) -> dict:
    global store
    if store is None:
        store = ChromaVectorStore(
            os.getenv("CHROMA_PATH", "./.chroma"),
            os.getenv("CHROMA_COLLECTION", "datheon_documents"),
            LocalSentenceEmbedder(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")),
        )
    chunks = retrieve(store, request.question, user, limit=request.limit)
    answer = grounded_answer(
        request.question,
        chunks,
        openai_client=openai_client,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    return {"answer": answer, "sources": citations_for(chunks)}