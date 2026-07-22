"""
Loads configuration for the tactics module from environment variables
(and a local .env file in development, via python-dotenv).

NEVER hardcode an API key here or anywhere else in this repo. Set
GEMINI_API_KEY as an environment variable locally (.env, gitignored)
and as a secret/environment variable on your host (Render, Railway,
etc.) in production.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
TACTICS_MAX_FRAMES = int(os.getenv("TACTICS_MAX_FRAMES", "8"))
TACTICS_MAX_EVENTS_IN_PROMPT = int(os.getenv("TACTICS_MAX_EVENTS_IN_PROMPT", "120"))


def require_api_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to backend/.env (local dev) "
            "or as an environment variable on your host (production)."
        )
    return GEMINI_API_KEY
