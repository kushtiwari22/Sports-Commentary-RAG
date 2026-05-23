# 🏏 Cricket AI — Live Commentary & Agentic Sports Chatbot

This project combines a live cricket commentary system with a sports question-answering chatbot.

- **Core backend** contributed: a structured cricket data pipeline, a FastAPI backend, and a Phidata-based commentary team (StatsAnalyzer + CommentaryGenerator powered by Groq/LLaMA).
- **Additional modules** contributed: a LangGraph agentic-RAG chatbot with Pathway vector store, Google Custom Search, multi-sport routing, and a Gradio UI.

---

## Architecture

```
User
 │
 ├── Gradio UI (ui.py)
 │      ├── Tab 1: Sports Q&A Chatbot  ──► POST /api/ask
 │      └── Tab 2: Live Cricket         ──► POST /api/set-match / GET /api/live-data
 │
 └── FastAPI Backend (main.py)
        │
        ├── /api/set-match/{id}
        │      └── CricketDataPipeline → starts local Flask cricket API
        │
        ├── /api/live-data
        │      └── CricketDataPipeline.fetch() → Phidata Commentary Team
        │             ├── StatsAnalyzer  (Groq/LLaMA + DuckDuckGo)
        │             └── CommentaryGenerator (Groq/LLaMA)
        │
        └── /api/ask
               └── LangGraph Agentic-RAG
                      ├── agent_decision (GPT-4o router)
                      │      ├── commentary_agent  → ScraperAPI diff → GPT-4o summary
                      │      ├── llm_agent         → GPT-4o direct answer
                      │      └── data_creation     → Google Search → scrape
                      │                               → Pathway VectorStore
                      │                                   → retrieve → grade
                      │                                   → generate → grade
                      │                                   → transform_query (loop)
                      └── Pathway VectorStoreServer (background thread)
```

---

## Quickstart

### 1. Clone & install

```bash
git clone <this-repo>
cd cricket_ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

Required keys:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o for routing, grading, generation, summaries |
| `GROQ_API_KEY` | LLaMA-3.3-70b via Groq for Phidata commentary team |
| `GOOGLE_API_KEY` | Google Custom Search API |
| `GOOGLE_CX_ID` | Your Custom Search Engine ID |
| `SCRAPER_API_KEY` | *(Optional)* ScraperAPI for JS-rendered pages |

### 3. Start the FastAPI backend

```bash
python main.py
# → listening on http://0.0.0.0:8080
```

### 4. Start the Gradio frontend (separate terminal)

```bash
python ui.py
# → opens browser / prints a share URL
```

---

## Endpoints

### `POST /api/set-match/{match_id}`

Set the active cricket match. This clones [sanwebinfo/cricket-api](https://github.com/sanwebinfo/cricket-api) if needed and starts the local Flask server.

```bash
curl -X POST http://localhost:8080/api/set-match/12345
```

### `GET /api/live-data`

Fetch live scorecard data and generate full AI commentary via the Phidata team.

```bash
curl http://localhost:8080/api/live-data
```

Response:
```json
{
  "match_id": "12345",
  "raw_data": { ... },
  "processed_data": { "team1": "MI", "team2": "KKR", "current_score": "185/4", ... },
  "commentary": "## Live Commentary\n\nMumbai Indians are cruising..."
}
```

### `POST /api/ask`

Multi-sport agentic Q&A chatbot.

```bash
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "MI vs KKR Live Commentary 2025", "input_days": 1}'
```

Response:
```json
{
  "question": "MI vs KKR Live Commentary 2025",
  "answer": "Mumbai Indians are at 45/2 after 6 overs...",
  "is_commentary": true
}
```

---

## Agent Routing Logic

```
User question
    │
    ▼
Does question ask for "Live Commentary"?
    ├── YES → commentary_agent
    │            ├── Classify sport (cricket / football / basketball)
    │            ├── Google search for live-scores page
    │            ├── Scrape page twice (5s apart), extract diff
    │            └── GPT-4o summary → structured commentary
    │
    └── NO → Can GPT-4o answer from training data?
                ├── YES → llm_agent (direct GPT-4o answer)
                └── NO  → data_creation_and_scraping
                              ├── Google Custom Search
                              ├── Scrape top 7 links via ScraperAPI
                              ├── Ingest into Pathway VectorStore
                              └── Retrieve → Grade → Generate → Grade (loop)
```

---

## How live cricket commentary works (Tab 2)

1. Find the match ID from the Cricbuzz URL, e.g. `https://www.cricbuzz.com/live-cricket-scores/**94567**/...` → ID is `94567`.
2. Enter it in the UI and click **Set Match**.
3. Click **Fetch Commentary** — the system fetches structured ball-by-ball data from the local Flask cricket-API, processes player stats, and generates rich commentary using the Phidata StatsAnalyzer + CommentaryGenerator team.

---

## Project Structure

```
cricket_ai/
├── main.py                  # FastAPI backend
├── ui.py                    # Gradio frontend
├── requirements.txt
├── .env.example
├── data/                    # Auto-created; holds scraped .txt files
└── src/
    ├── __init__.py
    ├── agents.py            # Phidata team + LangGraph graph (all agent logic)
    ├── cricket_pipeline.py  # CricketDataPipeline (Flask API wrapper)
    ├── rag_pipeline.py      # Pathway VectorStoreServer + retriever
    └── commentary_utils.py  # Web-scraping diff utilities
```

---

## Notes

- The Pathway VectorStore starts in a background thread the first time the `data_creation_and_scraping` node runs. It watches `./data/*.txt` and re-indexes automatically.
- ScraperAPI is optional but strongly recommended for JavaScript-rendered sports pages. Without it, plain `requests.get` is used (may miss dynamic content).
- The local Flask cricket-API (sanwebinfo/cricket-api) is cloned automatically on first use. It requires `flask` to be installed, which is included in `requirements.txt`.