

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import requests as http_requests

# Backend base URL — change if the FastAPI server runs elsewhere
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")


# Tab 1 — Agentic Sports Chatbot

def chatbot_ask(user_message: str, search_days: int, chat_history: list) -> tuple:
    """Send the question to /api/ask and append result to chat history."""
    if not user_message.strip():
        return chat_history, chat_history

    chat_history = chat_history or []
    chat_history.append(("👤 You", user_message))

    try:
        resp = http_requests.post(
            f"{BACKEND_URL}/api/ask",
            json={"question": user_message, "input_days": int(search_days)},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "No answer returned.")
        tag = "🎙️ Commentary" if data.get("is_commentary") else "🤖 AI"
        chat_history.append((tag, answer))
    except Exception as e:
        chat_history.append(("⚠️ Error", str(e)))

    return chat_history, chat_history


def clear_chat():
    return [], []


# Tab 2 — Live Cricket Match Commentary

def set_match(match_id: str) -> str:
    """Call /api/set-match and return status message."""
    if not match_id.strip():
        return "⚠️ Please enter a match ID."
    try:
        resp = http_requests.post(
            f"{BACKEND_URL}/api/set-match/{match_id.strip()}", timeout=60
        )
        resp.raise_for_status()
        return f"✅ {resp.json().get('message', 'Match set.')}"
    except Exception as e:
        return f"❌ Error: {e}"


def fetch_commentary() -> str:
    """Fetch live structured data + AI commentary from the backend."""
    try:
        resp = http_requests.get(f"{BACKEND_URL}/api/live-data", timeout=180)
        resp.raise_for_status()
        data = resp.json()

        if "message" in data and not data.get("commentary"):
            return data["message"]

        pd = data.get("processed_data", {})
        commentary = data.get("commentary", "")

        header = (
            f"## {pd.get('team1', '?')} vs {pd.get('team2', '?')}\n"
            f"**Score:** {pd.get('current_score', 'N/A')}  "
            f"**Batsmen:** {pd.get('batsman', 'N/A')}  "
            f"**Bowlers:** {pd.get('bowler', 'N/A')}\n\n"
            f"---\n\n"
        )
        return header + commentary
    except Exception as e:
        return f"❌ Error fetching live data: {e}"


# Build UI

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="Cricket AI") as demo:
    gr.HTML(
        "<h1 style='text-align:center;color:#00aaff;'>🏏 Cricket AI — Live Commentary & Sports Chatbot</h1>"
        "<p style='text-align:center;color:#888;'>Powered by Pathway · LangGraph · Phidata · GPT-4o · Groq</p>"
    )

    with gr.Tabs():
        with gr.TabItem("⚽ Sports Q&A Chatbot"):
            gr.Markdown(
                "Ask anything about cricket, football, or basketball. "
                "For **live commentary** include the phrase *'Live Commentary'* in your question."
            )

            chatbot_ui = gr.Chatbot(
                label="Chat History",
                bubble_full_width=False,
                height=480,
            )

            with gr.Row():
                days_input = gr.Number(
                    label="Search recency (days)", value=7, minimum=1, maximum=30
                )
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. MI vs KKR Live Commentary 2025",
                    scale=4,
                )
                send_btn = gr.Button("🚀 Send", variant="primary")

            clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")

            state = gr.State(value=[])

            def _submit(q, days, history):
                return chatbot_ask(q, days, history)

            question_input.submit(_submit, [question_input, days_input, state], [chatbot_ui, state])
            send_btn.click(_submit, [question_input, days_input, state], [chatbot_ui, state])
            clear_btn.click(clear_chat, [], [chatbot_ui, state])

        with gr.TabItem("🏏 Live Cricket Commentary"):
            gr.Markdown(
                "Enter a **match ID** from [cricbuzz](https://www.cricbuzz.com/) "
                "(found in the match URL), then click **Set Match**. "
                "After that, click **Fetch Commentary** to get AI-generated play-by-play."
            )

            with gr.Row():
                match_id_input = gr.Textbox(
                    label="Match ID", placeholder="e.g. 12345", scale=3
                )
                set_match_btn = gr.Button("🔗 Set Match", variant="primary")

            match_status = gr.Textbox(label="Status", interactive=False)

            fetch_btn = gr.Button("📡 Fetch Commentary", variant="secondary")
            commentary_output = gr.Markdown(label="Live Commentary")

            set_match_btn.click(set_match, [match_id_input], [match_status])
            fetch_btn.click(fetch_commentary, [], [commentary_output])


if __name__ == "__main__":
    demo.launch(share=True)