from fastapi import FastAPI
from pydantic import BaseModel

from app import graph

app = FastAPI(title="Agentic Research-RAG")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(q: Query):
    return graph.run(q.question)
