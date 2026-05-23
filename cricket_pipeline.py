

import subprocess
import time
import requests
from pathlib import Path


class CricketDataPipeline:
    """Manages the local cricket API server and fetches structured match data."""

    def __init__(self, match_id: str):
        self.match_id = match_id
        self.api_url = f"http://127.0.0.1:5000/score?id={self.match_id}"
        self.server_process = None

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start_api_server(self) -> None:
        """Clone (if needed) and start the local Flask cricket-api server."""
        repo_path = Path("cricket-api/api")

        if not repo_path.exists():
            print("Cloning cricket-api repository…")
            result = subprocess.run(
                ["git", "clone", "https://github.com/sanwebinfo/cricket-api"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr}")

        # Check if a server is already responding
        try:
            requests.get("http://127.0.0.1:5000/", timeout=1)
            print("Flask server is already running.")
            return
        except requests.exceptions.RequestException:
            pass

        print("Starting Flask server…")
        self.server_process = subprocess.Popen(
            ["flask", "--app", "index.py", "run", "--host=0.0.0.0", "--port=5000"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        for _ in range(30):
            try:
                response = requests.get("http://127.0.0.1:5000/", timeout=2)
                if response.status_code == 200:
                    print("Flask server started successfully.")
                    return
            except requests.exceptions.RequestException:
                time.sleep(1)

        raise TimeoutError("Flask server failed to start within 30 seconds.")

    def stop_api_server(self) -> None:
        """Terminate the Flask server if we started it."""
        if self.server_process:
            self.server_process.terminate()
            print("Flask server stopped.")

    # ------------------------------------------------------------------
    # Data fetching & processing
    # ------------------------------------------------------------------

    def fetch_cricket_data(self) -> dict:
        """Fetch raw JSON from the local cricket API."""
        try:
            response = requests.get(self.api_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Cricket API request failed: {e}")
            return {}

    def process_api_response(self, data: dict) -> dict:
        """Transform the raw API payload into a normalised document dict."""
        return {
            "match_id": self.match_id,
            "timestamp": int(time.time()),
            "current_score": data.get("livescore", ""),
            "context": data.get("update", ""),
            "team1": self._extract_team(data.get("title", ""), 0),
            "team2": self._extract_team(data.get("title", ""), 1),
            "batsman": f"{data.get('batterone', '')}/{data.get('battertwo', '')}",
            "bowler": f"{data.get('bowlerone', '')}/{data.get('bowlertwo', '')}",
            "runs": self._parse_runs(data.get("livescore", "0/0")),
            "wicket": "wicket" in data.get("context", "").lower(),
            "player_stats": {
                "batsmen": {
                    data.get("batterone", ""): {
                        "runs": data.get("batsmanonerun", 0),
                        "balls": data.get("batsmanoneball", 0),
                    },
                    data.get("battertwo", ""): {
                        "runs": data.get("batsmantworun", 0),
                        "balls": data.get("batsmantwoball", 0),
                    },
                },
                "bowlers": {
                    data.get("bowlerone", ""): {
                        "overs": data.get("bowleroneovers", 0),
                        "runs": data.get("bowleronerun", 0),
                        "wickets": data.get("bowleronewicket", 0),
                    }
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_team(title: str, index: int) -> str:
        teams = title.split("vs") if "vs" in title else ["Unknown", "Unknown"]
        return teams[index].split("-")[0].strip() if index < len(teams) else "Unknown"

    @staticmethod
    def _parse_runs(score: str) -> int:
        try:
            return int(score.split("/")[0])
        except (IndexError, ValueError):
            return 0