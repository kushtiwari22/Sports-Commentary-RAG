

from __future__ import annotations

import os
import time
import tiktoken
from typing import List

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo

from src.commentary_utils import commentary_link_assignment, live_commentary_diff
from src.rag_pipeline import get_retriever, start_vector_store

import openai as _openai

openai_client = _openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 1.  Phidata Commentary Team  (Project-1 logic)

def _build_phidata_team() -> Agent:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    model = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

    stats_analyzer = Agent(
        name="StatsAnalyzer",
        model=model,
        tools=[DuckDuckGo()],
        instructions=[
            """
            You are an expert cricket statistician and analyst. Analyse the provided
            cricket match data to:

            1. Summarise the current match situation and key statistics.
            2. Identify exceptional performances from batsmen (high strike rates, milestones).
            3. Highlight impressive bowling figures (economy rates, wicket-taking spells).
            4. Calculate partnership statistics and run-rate trends.
            5. Identify potential record-breaking performances or notable achievements.
            6. Detect important game-changing moments worth highlighting.
            7. Use web search for interesting statistics about the current batsmen/bowlers.

            The match data includes: score, run rates, batsmen stats (runs, balls, SR),
            bowler stats (overs, runs, wickets, economy), and recent commentary.

            Provide your analysis in structured format to help the commentary team
            understand key aspects of the current match situation.
            """
        ],
        markdown=True,
    )

    commentary_generator = Agent(
        name="CommentaryGenerator",
        model=model,
        instructions=[
            """
            You are an elite cricket commentator renowned for captivating, insightful,
            and engaging commentary.

            Using the match data and statistical analysis provided:

            1. Create vibrant play-by-play commentary that brings the cricket match to life.
            2. Weave statistical insights naturally into your narrative.
            3. Use colourful language, cricket terminology, and appropriate expressions.
            4. Vary your tone to match the game situation — excited for boundaries,
               analytical for strategy.
            5. Incorporate the latest match developments from the 'context' field.
            6. Build upon but don't simply repeat existing commentary.
            7. Include both immediate action description and strategic analysis.

            Create at least 250 words of rich, compelling cricket commentary.
            """
        ],
        markdown=True,
    )

    return Agent(
        name="Cricket Commentary Team",
        team=[stats_analyzer, commentary_generator],
        model=model,
        instructions=[
            "Coordinate the team to generate engaging cricket commentary.",
            "First have StatsAnalyzer analyse the match data and identify highlights.",
            "Then have CommentaryGenerator use the analysis to create engaging commentary.",
            "Ensure the final output is cohesive and engaging.",
        ],
        markdown=True,
    )


_commentary_team: Agent | None = None


def get_commentary_team() -> Agent:
    global _commentary_team
    if _commentary_team is None:
        _commentary_team = _build_phidata_team()
    return _commentary_team


def generate_cricket_commentary(processed_data: dict) -> str:
    """
    Run the Phidata cricket commentary team on pre-processed match data.
    Returns the commentary as a markdown string.
    """
    team = get_commentary_team()
    result = team.run(cricket_data=processed_data)
    # Phidata returns a RunResponse; extract the content string
    if hasattr(result, "content"):
        return result.content or str(result)
    return str(result)


# 2.  LangGraph Agentic-RAG  (Project-2 logic, enhanced)

_llm = ChatOpenAI(model="gpt-4o", temperature=0)

_rag_prompt = hub.pull("rlm/rag-prompt")
_rag_chain = _rag_prompt | _llm | StrOutputParser()



class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="'yes' or 'no'")


_doc_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing relevance of a retrieved document to a user question. "
                "Give binary score 'yes' or 'no'. Be lenient — 'no' only in very harsh scenarios.",
            ),
            ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
        ]
    )
    | _llm.with_structured_output(GradeDocuments)
)

_hallucination_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an LLM generation is grounded in the retrieved facts. "
                "Give binary 'yes' (grounded) or 'no'. Be lenient.",
            ),
            ("human", "Facts:\n\n{documents}\n\nGeneration: {generation}"),
        ]
    )
    | _llm.with_structured_output(GradeHallucinations)
)

_answer_grader = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a grader assessing whether an answer resolves the question. "
                "Give binary 'yes' or 'no'.",
            ),
            ("human", "Question:\n\n{question}\n\nAnswer: {generation}"),
        ]
    )
    | _llm.with_structured_output(GradeAnswer)
)

_query_rewriter = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite questions for vectorstore retrieval. "
                "Make them short and semantically precise. "
                "E.g. 'Virat Kohli T20 stats' not 'What are the statistics of Virat Kohli in T20 cricket?'",
            ),
            ("human", "Question: {question}\n\nFormulate an improved question."),
        ]
    )
    | _llm
    | StrOutputParser()
)



def _query_type_answer(question: str) -> str:
    """Return 'yes' if the query is asking for live commentary."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Determine if the query is asking for live cricket/sports commentary. "
                    "Respond ONLY 'yes' or 'no'. "
                    "Only 'yes' if the phrase 'live commentary' or equivalent is present."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


def _query_answer_from_memory(question: str) -> str:
    """Return 'yes' if GPT-4o can answer the question from training data alone."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a sports AI. Determine if the query can be answered from "
                    "your training data (historical, factual, pre-Oct 2023). "
                    "Answer ONLY 'yes' or 'no'. "
                    "Say 'no' for live scores, current stats of active players, "
                    "recent matches, or subjective questions."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


def _query_sport(question: str) -> str:
    """Classify the query sport: 'football', 'cricket', or 'basketball'."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the sport in the query. "
                    "Respond ONLY with one word: 'football', 'cricket', or 'basketball'."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip().lower()


def _query_constructor(question: str) -> str:
    """Rewrite the query to be optimal for Google Custom Search."""
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the query for concise Google search. "
                    "E.g. 'Virat Kohli T20 stats latest'. "
                    "Exclude: -site:x.com -site:youtube.com -site:reddit.com "
                    "-site:instagram.com -site:facebook.com -site:linkedin.com"
                ),
            },
            {"role": "user", "content": f"Rewrite for web search: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def google_search(query: str, search_days: str) -> tuple[list[str], list[str]]:
    """Query Google Custom Search API; returns (links, snippets)."""
    google_api_key = os.environ["GOOGLE_API_KEY"]
    cx_id = os.environ["GOOGLE_CX_ID"]
    url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?q={query}&key={google_api_key}&cx={cx_id}&dateRestrict=d{search_days}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print(f"Google search error: {e}")
        return [], []

    links = [item["link"] for item in results.get("items", [])]
    snippets = [item.get("snippet", "") for item in results.get("items", [])]
    return links, snippets


import requests  # noqa: E402  (needed inside google_search above)



def summary_agent(txt_file: str, question: str, model: str = "gpt-4o", batch_size: int = 100_000) -> str:
    """
    Summarise a (potentially large) text file with respect to `question`.
    Processes in token batches and combines into a final match commentary.
    """
    enc = tiktoken.encoding_for_model(model)
    data = open(txt_file, encoding="utf-8").read()
    tokens = enc.encode(data)

    batches = [tokens[i: i + batch_size] for i in range(0, len(tokens), batch_size)]
    batch_summaries: list[str] = []

    for i, batch in enumerate(batches):
        text = enc.decode(batch)
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a sports AI. Summarise the following live-sports text "
                        f"(batch {i + 1} of {len(batches)}):\n\n{text}"
                    ),
                }
            ],
        )
        batch_summaries.append(resp.choices[0].message.content)

    final_resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate concise live commentary for the match in '{question}' "
                    f"based on these batch summaries. Focus only on the latest updates.\n\n"
                    + "\n\n---\n\n".join(batch_summaries)
                ),
            }
        ],
    )
    return final_resp.choices[0].message.content



class GraphState(TypedDict):
    question: str
    input_days: str
    generation: str
    documents: List[str]
    commentary: int  # 1 = commentary was generated; 0 = standard Q&A



def agent_decision(state: GraphState) -> str:
    question = state["question"]
    if _query_type_answer(question) == "yes":
        print("[Router] → commentary_agent")
        return "commentary_agent"
    if _query_answer_from_memory(question) == "yes":
        print("[Router] → llm_agent")
        return "llm_agent"
    print("[Router] → data_creation_and_scraping")
    return "data_creation_and_scraping"


def commentary_agent(state: GraphState) -> GraphState:
    """
    Handles live commentary requests.
    • For cricket: tries the structured CricketDataPipeline first (Phidata team).
    • For all sports: falls back to web-scraping diff + summary_agent.
    """
    question = state["question"]
    sport = _query_sport(question)

    # -- Select target site --
    if sport == "football":
        links, _ = google_search(
            _query_constructor(f"site:https://www.sofascore.com/football/match/ {question}"), "1"
        )
    elif sport == "cricket":
        links, _ = google_search(
            f"site:www.cricbuzz.com/live-cricket-scores/ {question}", "1"
        )
    else:
        links, _ = google_search(
            _query_constructor(f"site:https://www.espn.com/ {question}"), "1"
        )

    link = commentary_link_assignment(question, links, openai_client) if links else ""

    if link:
        diff_text = live_commentary_diff(link, sleep_duration=5)
        generated = summary_agent("./data/diff.txt", question)
    else:
        generated = f"Could not find a live page for: {question}"

    return {**state, "generation": generated, "commentary": 1}


def llm_agent(state: GraphState) -> GraphState:
    question = state["question"]
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert sports knowledge AI."},
            {"role": "user", "content": question},
        ],
    )
    return {**state, "generation": resp.choices[0].message.content}


def data_creation_and_scraping(state: GraphState) -> GraphState:
    """Google-search → scrape top links → ingest into Pathway vector store."""
    from src.rag_pipeline import ingest_text_file, start_vector_store

    question = state["question"]
    search_days = state.get("input_days", "7")

    links, _ = google_search(_query_constructor(question), str(search_days))
    while not links:
        links, _ = google_search(_query_constructor(question), str(search_days))

    scraper_api_key = os.getenv("SCRAPER_API_KEY", "")
    for i, link in enumerate(links[:7], start=1):
        try:
            if scraper_api_key:
                r = requests.get(
                    "https://api.scraperapi.com/",
                    params={"api_key": scraper_api_key, "url": link, "device_type": "desktop"},
                    timeout=30,
                )
            else:
                r = requests.get(link, timeout=30)
            ingest_text_file(r.text, str(i))
        except Exception as e:
            print(f"Scraping failed for {link}: {e}")

    # (Re-)start Pathway vector store to pick up newly ingested files
    start_vector_store()
    return state


def retrieve(state: GraphState) -> GraphState:
    question = state["question"]
    rewritten = _query_rewriter.invoke({"question": question})
    try:
        retriever = get_retriever()
        docs = retriever.invoke(rewritten)
    except Exception as e:
        print(f"Retrieval error: {e}")
        docs = []
    return {**state, "documents": [d.page_content for d in docs]}


def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    filtered: list[str] = []
    for doc in state.get("documents", []):
        score = _doc_grader.invoke({"document": doc, "question": question})
        if score.binary_score.lower() == "yes":
            filtered.append(doc)
    return {**state, "documents": filtered}


def decide_to_generate(state: GraphState) -> str:
    return "generate" if state.get("documents") else "transform_query"


def generate(state: GraphState) -> GraphState:
    docs_text = "\n\n".join(state.get("documents", []))
    generation = _rag_chain.invoke(
        {"context": docs_text, "question": state["question"]}
    )
    return {**state, "generation": generation}


def transform_query(state: GraphState) -> GraphState:
    new_q = _query_rewriter.invoke({"question": state["question"]})
    return {**state, "question": new_q}


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    docs_text = "\n\n".join(state.get("documents", []))
    hallucination = _hallucination_grader.invoke(
        {"documents": docs_text, "generation": state["generation"]}
    )
    if hallucination.binary_score.lower() == "yes":
        answer_grade = _answer_grader.invoke(
            {"question": state["question"], "generation": state["generation"]}
        )
        return "useful" if answer_grade.binary_score.lower() == "yes" else "not useful"
    return "not supported"


# 3.  Build & compile the LangGraph

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("commentary_agent", commentary_agent)
    workflow.add_node("llm_agent", llm_agent)
    workflow.add_node("data_creation_and_scraping", data_creation_and_scraping)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_conditional_edges(
        START,
        agent_decision,
        {
            "commentary_agent": "commentary_agent",
            "llm_agent": "llm_agent",
            "data_creation_and_scraping": "data_creation_and_scraping",
        },
    )
    workflow.add_edge("commentary_agent", END)
    workflow.add_edge("llm_agent", END)
    workflow.add_edge("data_creation_and_scraping", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph