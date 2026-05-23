

import difflib
import os
import time
from pathlib import Path

import requests

DATA_PATH = Path("./data")
DATA_PATH.mkdir(exist_ok=True)

SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scrape_url(url: str) -> str:
    """Fetch a page via ScraperAPI; falls back to direct GET if no key set."""
    if SCRAPER_API_KEY:
        payload = {
            "api_key": SCRAPER_API_KEY,
            "url": url,
            "device_type": "desktop",
        }
        r = requests.get("https://api.scraperapi.com/", params=payload, timeout=30)
    else:
        r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def _write(text: str, name: str) -> Path:
    path = DATA_PATH / f"scraped_data_{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def live_commentary_diff(url: str, sleep_duration: int = 5) -> str:
    """
    Fetch a live-scores page twice, `sleep_duration` seconds apart,
    and return only the *new* lines (the diff).

    Returns:
        A string of newly added lines – the live update delta.
    """
    old_text = _scrape_url(url)
    _write(old_text, "old")

    time.sleep(sleep_duration)

    new_text = _scrape_url(url)
    _write(new_text, "new")

    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)

    diff = difflib.unified_diff(old_lines, new_lines, lineterm="")
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]

    diff_text = "\n".join(added)
    (DATA_PATH / "diff.txt").write_text(diff_text, encoding="utf-8")
    return diff_text


def commentary_link_assignment(question: str, links: list[str], openai_client) -> str:
    """Ask GPT-4o to pick the best link from a list for the given question."""
    if not links:
        return ""

    system_prompt = (
        "You are an AI agent specialising in sports knowledge. "
        "You will be given a list of URLs and a query. "
        "Output only the single most relevant URL — no extra text.\n\n"
        f"Links:\n" + "\n".join(links)
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()