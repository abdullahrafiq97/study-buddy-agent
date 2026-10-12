# tutor/cli.py
"""Take the quiz in the terminal and watch the agent's trace live.

    python -m tutor.cli                 # agent decides, asks for your reason after wrong answers
    python -m tutor.cli --rules         # same quiz, hand-written rules decide (for comparison)
    python -m tutor.cli --no-reasons    # don't ask for reasons
"""
import argparse

from . import live, usage
from .agent import AgentError, AgentPolicy
from .concepts import CONCEPTS
from .policies import HeuristicPolicy
from .session import Session

CONF = {"s": "sure", "u": "unsure", "g": "guess"}


def ask(prompt: str, valid) -> str:
    while True:
        v = input(prompt).strip().lower()
        if v in valid:
            return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="hand-written rules decide instead of the LLM agent")
    ap.add_argument("--no-reasons", action="store_true", help="don't ask why after wrong answers")
    ap.add_argument("--max-q", type=int, default=12)
    ap.add_argument("--concepts", nargs="+", choices=list(CONCEPTS), help="limit the session to these concepts")
    args = ap.parse_args()

    policy = HeuristicPolicy() if args.rules else AgentPolicy()
    session = Session(policy, concepts=args.concepts, max_questions=args.max_q,
                      min_questions=min(6, args.max_q), reasons=not args.no_reasons)
    try:
        while True:
            if session.ctx.answered < session.ctx.max_questions:
                live.deciding(session.ctx.answered + 1, policy.name)
            step = session.next_question()
            if step is None:
                if session.trace and session.trace[-1].decision.action == "stop":
                    live.decision(session.trace[-1])  # show why the agent chose to stop
                break
            live.decision(step)
            live.question(step)
            idx = "abcd".index(ask("\n   Your answer (A-D): ", "abcd"))
            conf = CONF[ask("   Confidence: [s]ure, [u]nsure, [g]uess: ", CONF)]
            reason = None
            if session.wants_reason(idx):
                reason = input(f"   Before I show the answer: why did you pick {'ABCD'[idx]}? ").strip() or None
                if reason:
                    print(f"{live.DIM}   reading your reason (API call)...{live.RESET}", flush=True)
            live.answered(session.submit(idx, conf, reason))
    except AgentError as e:
        print(f"\n{live.RED}LLM call failed: {e}{live.RESET}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    if session.beliefs.answers:
        live.finished(session)
    print("\nLLM usage and cost:\n" + usage.report())


if __name__ == "__main__":
    main()
