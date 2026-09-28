# tutor/usage.py
"""Token and cost tracking for every LLM call, so each run reports what it spent."""
import threading
from collections import defaultdict

# USD per 1M tokens (input, cached input, output), Standard tier, from
# developers.openai.com/api/docs/pricing (checked 2026-09-28). Reasoning tokens bill as output.
PRICES = {
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
}

_lock = threading.Lock()
_totals = defaultdict(lambda: {"calls": 0, "input": 0, "cached": 0, "output": 0})


def record(role: str, model: str, usage) -> None:
    if usage is None:
        return
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    with _lock:
        t = _totals[(role, model)]
        t["calls"] += 1
        t["input"] += usage.prompt_tokens
        t["cached"] += cached
        t["output"] += usage.completion_tokens


def cost(model: str, t: dict) -> float:
    p = PRICES.get(model)
    if p is None:
        return float("nan")
    return ((t["input"] - t["cached"]) * p[0] + t["cached"] * p[1] + t["output"] * p[2]) / 1e6


def report() -> str:
    if not _totals:
        return "No LLM calls."
    lines = ["| Role | Model | Calls | Input tok/call | Output tok/call | Cost |", "|---|---|---|---|---|---|"]
    total = 0.0
    for (role, model), t in sorted(_totals.items()):
        c = cost(model, t)
        total += c
        lines.append(f"| {role} | {model} | {t['calls']} | {t['input'] // t['calls']} | "
                     f"{t['output'] // t['calls']} | ${c:.3f} |")
    lines.append(f"\nTotal: ${total:.3f}")
    return "\n".join(lines)
