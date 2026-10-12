# tutor/live.py
"""Print the agent's trace to the terminal as it happens, one step at a time."""
import textwrap

from .concepts import CONCEPTS, MISCONCEPTIONS
from .render import ACTION_LABELS, LETTERS

BOLD, DIM, CYAN, YELLOW, GREEN, RED, MAGENTA, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[32m", "\033[31m", "\033[35m", "\033[0m")


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def deciding(number: int, policy_name: str):
    who = "agent is deciding (API call)" if policy_name.startswith("agent") else f"{policy_name} policy is deciding"
    print(f"\n{DIM}━━━━ step {number}: {who}...{RESET}", flush=True)


def decision(step):
    d = step.decision
    head = f"{BOLD}{CYAN}DECISION #{step.number} [{d.policy}]: {ACTION_LABELS.get(d.action, d.action)}{RESET}"
    if d.concept:
        head += f"{CYAN} → {CONCEPTS[d.concept].name}, level {d.level}{RESET}"
        if d.target:
            head += f"{CYAN}, testing '{d.target}'{RESET}"
    print(head)
    for n in d.guardrail_notes:
        print(f"{YELLOW}   ⚠ guardrail: {n}{RESET}")
    if d.hypothesis:
        print(f"{BOLD}   read so far:{RESET}")
        print(_wrap(d.hypothesis, "     "))
    print(f"{BOLD}   why this move:{RESET}")
    print(_wrap(d.rationale, "     "))
    if d.if_correct:
        print(f"{DIM}   if correct → {d.if_correct}{RESET}")
    if d.if_wrong:
        print(f"{DIM}   if wrong   → {d.if_wrong}{RESET}")


def question(step):
    q = step.question
    print(f"\n{BOLD}QUESTION {step.number}{RESET} {DIM}({q.id}, {CONCEPTS[q.concept].name}, L{q.level}, {q.source}){RESET}")
    print(_wrap(q.stem, "   "))
    for i, o in enumerate(q.options):
        print(_wrap(f"{LETTERS[i]}. {o.text}", "     "))


def answered(step):
    q = step.question
    chosen = q.options[step.chosen_index]
    if step.correct:
        print(f"{GREEN}   ✓ {LETTERS[step.chosen_index]} is correct ({step.confidence}){RESET}")
    else:
        tag = f", the option built on '{chosen.misconception}'" if chosen.misconception else ", an untagged wrong option"
        print(f"{RED}   ✗ {LETTERS[step.chosen_index]} is wrong ({step.confidence}{tag}); answer is {LETTERS[q.correct_index]}{RESET}")
    if step.reading is not None:
        r = step.reading
        label = r.kind + (f": {r.misconception_id}" if r.misconception_id else "")
        print(f"{MAGENTA}   reason: \"{r.text}\"{RESET}")
        print(f"{MAGENTA}   model read it as → {label}. {r.explanation}{RESET}")
    print(f"{DIM}   updated beliefs:{RESET}")
    for n in step.notes:
        print(f"{DIM}     · {n}{RESET}")


def finished(session):
    diag = session.diagnosis()
    print(f"\n{BOLD}━━━━ SESSION OVER after {len(session.beliefs.answers)} questions{RESET}")
    print(_wrap(session.stop_reason, "   "))
    print(f"{BOLD}DIAGNOSIS{RESET}")
    for c, label in diag.labels.items():
        cb = session.beliefs.concept_beliefs[c]
        print(f"   {CONCEPTS[c].name:<40} {label:<14} mastery {cb.mean:.2f} ± {cb.sd:.2f}")
    for m in diag.confirmed:
        print(f"{RED}   misconception found: {MISCONCEPTIONS[m].description}{RESET}")
    for m in diag.suspected:
        print(f"{YELLOW}   possible misconception (seen once): {MISCONCEPTIONS[m].description}{RESET}")
    for c, p in diag.root_causes:
        print(f"{YELLOW}   likely root cause: {CONCEPTS[c].name} is weak and so is its prerequisite {CONCEPTS[p].name}{RESET}")
