# tutor/config.py
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.4-mini")
# Reasoning effort for GPT-5-family models ("minimal", "low", "medium", "high").
AGENT_REASONING = os.getenv("AGENT_REASONING", "low")
