

import os
import sys

# Ensure project root is on the path so `src.*` imports work
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.cricket_pipeline import CricketDataPipeline
from src.agents import generate_cricket_commentary, get_graph

app = FastAPI(
    title="Cricket AI — Agentic Commentary & RAG Chatbot",
    description=(
        "Merged system: Phidata cricket commentary team "
        "+ LangGraph agentic-RAG for multi-sport Q&A."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: CricketDataPipeline | None = None


class AskRequest(BaseModel):
    question: str
    input_days: int = 7  # how many days back to restrict Google search results


class AskResponse(BaseModel):
    question: str
    answer: str
    is_commentary: bool



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/set-match/{match_id}")
def set_match(match_id: str):
    """Start the local Flask cricket-API and pin it to `match_id`."""
    global _pipeline
    try:
        pipeline = CricketDataPipeline(match_id)
        pipeline.start_api_server()
        _pipeline = pipeline
        return {"message": f"Match ID set to '{match_id}'. Cricket API server started."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/live-data")
def get_live_data():
    """
    Fetch structured cricket data for the active match and generate
    AI commentary using the Phidata StatsAnalyzer + CommentaryGenerator team.
    """
    global _pipeline
    if _pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="No active match. Call POST /api/set-match/{match_id} first.",
        )
    try:
        raw = _pipeline.fetch_cricket_data()
        if not raw:
            return {"message": "No data returned from cricket API.", "data": {}}

        processed = _pipeline.process_api_response(raw)

        print("[/api/live-data] Running Phidata commentary team…")
        commentary = generate_cricket_commentary(processed)

        return {
            "match_id": _pipeline.match_id,
            "raw_data": raw,
            "processed_data": processed,
            "commentary": commentary,
        }
    except Exception as exc:
        print(f"[/api/live-data] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/ask", response_model=AskResponse)
def ask(body: AskRequest):
    """
    Agentic-RAG sports chatbot.
    Routes to: live-commentary scraper | LLM-only answer | Pathway RAG pipeline.
    """
    graph = get_graph()
    try:
        result = graph.invoke(
            {
                "question": body.question,
                "input_days": str(body.input_days),
                "generation": "",
                "documents": [],
                "commentary": 0,
            }
        )
        return AskResponse(
            question=body.question,
            answer=result.get("generation", "No answer generated."),
            is_commentary=bool(result.get("commentary", 0)),
        )
    except Exception as exc:
        print(f"[/api/ask] Error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)