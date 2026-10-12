# tutor/simulate.py
"""
Verification with simulated students.

Each simulated student has a known ground truth: concept mastery levels and
planted misconceptions. We run full sessions with each policy and check whether
the final diagnosis finds what we planted, without false alarms.

Usage:
    python -m tutor.simulate                       # rule + random baselines, 10 runs per student
    python -m tutor.simulate --policies agent rule random --runs 5
    python -m tutor.simulate --show weak_foundations --policies agent   # print one full trace
"""
import argparse
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .bank import Question, load_bank
from .concepts import CONCEPTS, MISCONCEPTIONS, dependents
from .policies import HeuristicPolicy, RandomPolicy, RuleBasedPolicy
from .session import Session

KNOWS_CORRECT = {1: 0.95, 2: 0.88, 3: 0.75}  # P(correct | knows the concept), by level
FALLS_FOR_HELD = 0.85                          # P(picks the option for a misconception they hold)
CONFIDENCE_NOISE = 0.0  # P(the reported confidence is random); real students are often miscalibrated


@dataclass
class SimStudent:
    name: str
    description: str
    mastery: Dict[str, float] = field(default_factory=dict)
    misconceptions: Tuple[str, ...] = ()

    def m(self, concept: str) -> float:
        return self.mastery.get(concept, 0.9)

    def effective_mastery(self, concept: str) -> float:
        k = self.m(concept)
        for p in CONCEPTS[concept].prereqs:
            k *= min(1.0, 0.3 + self.m(p))  # a weak prerequisite drags this concept down
        return k

    def answer(self, q: Question, rng: random.Random) -> Tuple[int, str, str]:
        """Returns (option index, confidence, true cause). The cause is hidden from
        the system; it is used to write a realistic reason and to score the reader."""
        idx, conf, cause = self._answer(q, rng)
        # Only draw when noise is on, so noise-free runs keep the same random stream (reproducible results).
        if CONFIDENCE_NOISE and rng.random() < CONFIDENCE_NOISE:
            conf = rng.choice(["sure", "unsure", "guess"])
        return idx, conf, cause

    def _answer(self, q: Question, rng: random.Random) -> Tuple[int, str, str]:
        held = [i for i, o in enumerate(q.options) if o.misconception in self.misconceptions]
        if held and rng.random() < FALLS_FOR_HELD:
            i = rng.choice(held)
            return i, ("sure" if rng.random() < 0.7 else "unsure"), f"misconception:{q.options[i].misconception}"
        wrong = [i for i, o in enumerate(q.options) if not o.correct]
        if rng.random() < self.effective_mastery(q.concept):
            if rng.random() < KNOWS_CORRECT[q.level]:
                conf = "sure" if q.level < 3 or rng.random() < 0.6 else "unsure"
                return q.correct_index, conf, "knows"
            return rng.choice(wrong), "unsure", "slip"
        return rng.randrange(4), ("guess" if rng.random() < 0.7 else "unsure"), "guess"

    def truth(self, concepts) -> dict:
        weak = {c for c in concepts if self.m(c) < 0.5}
        downstream = {d for c in weak for d in dependents(c)}
        return {"weak": weak, "misconceptions": set(self.misconceptions), "ambiguous": downstream - weak}


PERSONAS: List[SimStudent] = [
    SimStudent("strong_all_round", "Solid on everything; should get no false alarms."),
    SimStudent("swaps_sens_spec", "Knows the material but swaps sensitivity and specificity.",
               misconceptions=("sensitivity_specificity_swap",)),
    SimStudent("prob_not_logodds", "Reads logit coefficients as probability changes.",
               mastery={"logistic": 0.75}, misconceptions=("logit_coef_is_prob_change",)),
    SimStudent("trusts_training_error", "Picks models by training error.",
               mastery={"overfitting": 0.7}, misconceptions=("train_error_is_enough",)),
    SimStudent("shrinkage_gap", "Never learned ridge/LASSO; no specific misconception.",
               mastery={"model_selection": 0.1}),
    SimStudent("weak_foundations", "Shaky on OLS basics, which drags down everything built on it.",
               mastery={"ols_basics": 0.15}),
    SimStudent("two_misconceptions", "Reads dummies as group means and has λ backwards.",
               misconceptions=("dummy_is_group_mean", "bigger_lambda_less_shrinkage")),
    SimStudent("classification_struggler", "Weak on classification and trusts accuracy on rare events.",
               mastery={"classification": 0.15}, misconceptions=("accuracy_on_rare_events",)),
]
PERSONA_BY_NAME = {p.name: p for p in PERSONAS}


UNHELPFUL_REASONS = ["idk", "seemed right", "not sure tbh", "it just looked correct", "?"]
UNHELPFUL_RATE = 0.2  # real students often give reasons that say nothing

STUDENT_PROMPT = """You are role-playing a university student taking a quiz on regression and classification.
You just picked a WRONG answer. Write the one short line (5-25 words) you would type when asked
"why did you pick that?". Write casually, in first person, like a real student. Never use the
phrase 'misconception', never quote the option text word for word, and don't say it's wrong
(you don't know yet). The hidden truth about why you picked it:
{cause}"""


def _cause_text(cause: str) -> str:
    if cause.startswith("misconception:"):
        m = MISCONCEPTIONS[cause.split(":", 1)[1]]
        return ("You sincerely hold this belief and think it is TRUE: " + m.description + "\n"
                "Justify the option you picked using that belief, in your own casual words. "
                "Argue FOR your pick; do not mention any doubts or the correct view.")
    if cause == "slip":
        return ("You understand this topic well and would normally get it right. This time you rushed: you skimmed "
                "the question or clicked the wrong letter. Say that you rushed or misread. Do NOT describe any "
                "belief about the topic.")
    return ("You don't understand this topic at all and picked at random. Say you didn't know or guessed. "
            "Do NOT give a specific technical justification.")


def student_reason(q: Question, idx: int, cause: str, rng: random.Random) -> str:
    """A simulated student's one-line explanation, written by an LLM in the student's voice."""
    if rng.random() < UNHELPFUL_RATE:
        return rng.choice(UNHELPFUL_REASONS)
    from openai import OpenAI
    from . import config
    client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=8)
    opts = "\n".join(f"{'ABCD'[i]}. {o.text}" for i, o in enumerate(q.options))
    resp = client.chat.completions.create(
        model=config.STUDENT_MODEL,
        messages=[{"role": "system", "content": STUDENT_PROMPT.format(cause=_cause_text(cause))},
                  {"role": "user", "content": f"{q.stem}\n\n{opts}\n\nYou picked {'ABCD'[idx]}: "
                                              f"\"{q.options[idx].text}\". Your one line:"}],
        **({"reasoning_effort": "low"} if config.STUDENT_MODEL.startswith("gpt-5") else {}),
    )
    from . import usage
    usage.record("simulated student", config.STUDENT_MODEL, resp.usage)
    return (resp.choices[0].message.content or "idk").strip().strip('"')


class OracleReader:
    """A perfect reason reader for the free ceiling test: it is handed the true cause
    (as the 'reason' text) and returns the ideal reading. No LLM calls."""

    def read(self, q, chosen_index, reason):
        from .reasons import ReasonReading
        if reason in UNHELPFUL_REASONS:
            return ReasonReading("unclear", None, "weak", "", "unhelpful reason", reason)
        if reason.startswith("misconception:"):
            return ReasonReading("misconception", reason.split(":", 1)[1], "strong", "", "oracle", reason)
        return ReasonReading(reason, None, "strong", "", "oracle", reason)  # "slip" or "guess"


def reading_matches(cause: str, reading) -> bool:
    """Did the reader recover the true cause?"""
    if cause.startswith("misconception:"):
        return reading.kind == "misconception" and reading.misconception_id == cause.split(":", 1)[1]
    if cause == "slip":
        return reading.kind == "slip"
    return reading.kind in ("guess", "gap")


def make_policy(name: str):
    name = name.replace("+reasons", "")
    if name == "agent":
        from .agent import AgentPolicy
        return AgentPolicy()
    return {"rule": RuleBasedPolicy, "random": RandomPolicy, "heuristic": HeuristicPolicy}[name]()


def session_events(policy_name: str, student: SimStudent, run: int, max_q: int, min_q: int, bank,
                   oracle: bool = False, concepts=None):
    """Run one session, yielding each step as it happens:
        ("deciding", number) → ("decision", step) → ("answered", step, true_cause) → ... → ("done", result)
    The terminal (--show), the app and the evaluation all run sessions through this."""
    reasons = policy_name.endswith("+reasons")
    session = Session(make_policy(policy_name), concepts=concepts, max_questions=max_q, min_questions=min_q,
                      seed=1000 + run, bank=bank, reasons=reasons,
                      reader=OracleReader() if oracle and reasons else None)
    rng = random.Random(f"{student.name}-{run}")
    reason_rng = random.Random(f"{student.name}-{run}-reasons")
    readings = []  # (true cause, reading) for scoring the reader
    while True:
        if session.ctx.answered < session.ctx.max_questions:
            yield ("deciding", session.ctx.answered + 1)
        step = session.next_question()
        if step is None:
            if session.trace and session.trace[-1].decision.action == "stop":
                yield ("decision", session.trace[-1])  # why the agent chose to stop
            break
        yield ("decision", step)
        idx, conf, cause = student.answer(step.question, rng)
        reason = None
        if session.wants_reason(idx):
            if oracle:
                reason = (reason_rng.choice(UNHELPFUL_REASONS) if reason_rng.random() < UNHELPFUL_RATE
                          else cause)
            else:
                reason = student_reason(step.question, idx, cause, reason_rng)
        session.submit(idx, conf, reason)
        yield ("answered", step, cause)
        if step.reading is not None:
            readings.append((cause, step.reading))

    diag = session.diagnosis()
    truth = student.truth(session.beliefs.concepts)
    flagged_weak, flagged_mis = set(diag.weak), set(diag.confirmed)
    result = {
        "policy": policy_name,
        "student": student.name,
        "run": run,
        "questions": len(session.beliefs.answers),
        "stopped_early": len(session.beliefs.answers) < max_q,
        "mis_tp": len(flagged_mis & truth["misconceptions"]),
        "mis_fp": len(flagged_mis - truth["misconceptions"]),
        "mis_fn": len(truth["misconceptions"] - flagged_mis),
        "weak_tp": len(flagged_weak & truth["weak"]),
        "weak_fp": len(flagged_weak - truth["weak"] - truth["ambiguous"]),
        "weak_fn": len(truth["weak"] - flagged_weak),
        "fallbacks": sum(1 for s in session.trace if s.decision.policy == "fallback"),
        "rejections": sum(len([n for n in s.decision.guardrail_notes if n.startswith("Rejected")])
                          for s in session.trace),
        "flagged_weak": sorted(flagged_weak),
        "flagged_misconceptions": sorted(flagged_mis),
        "suspected": sorted(diag.suspected),
        "stop_reason": session.stop_reason,
        "readings": [{"cause": c, "kind": r.kind, "misconception": r.misconception_id, "text": r.text,
                      "match": reading_matches(c, r)} for c, r in readings],
        "trace": [trace_to_dict(s) for s in session.trace],
    }
    result["exact"] = all(result[k] == 0 for k in ("mis_fp", "mis_fn", "weak_fp", "weak_fn"))
    result["session"] = session  # for live summaries; dropped before saving to JSON
    yield ("done", result)


def run_one(policy_name: str, student: SimStudent, run: int, max_q: int, min_q: int, bank,
            oracle: bool = False, stream: bool = False, concepts=None) -> dict:
    """Run a whole session. stream=True prints each step to the terminal as it happens (tutor/live.py)."""
    from . import live
    for event in session_events(policy_name, student, run, max_q, min_q, bank, oracle, concepts):
        kind = event[0]
        if kind == "done":
            result = event[1]
            if not stream:
                result.pop("session")
            return result
        if not stream:
            continue
        if kind == "deciding":
            live.deciding(event[1], policy_name)
        elif kind == "decision":
            live.decision(event[1])
            if event[1].question is not None:
                live.question(event[1])
        elif kind == "answered":
            print(f"{live.DIM}   simulated student answered (hidden cause: {event[2]}){live.RESET}")
            live.answered(event[1])


def trace_to_dict(s) -> dict:
    d = s.decision
    return {
        "n": s.number,
        "policy": d.policy, "action": d.action, "concept": d.concept, "level": d.level, "target": d.target,
        "hypothesis": d.hypothesis, "rationale": d.rationale,
        "if_correct": d.if_correct, "if_wrong": d.if_wrong,
        "guardrail_notes": d.guardrail_notes,
        "question": s.question.id if s.question else None,
        "correct": s.correct, "confidence": s.confidence,
        "picked_misconception": (s.question.options[s.chosen_index].misconception
                                 if s.question and s.chosen_index is not None else None),
        "reason": s.reading.text if s.reading else None,
        "reading": s.reading.kind if s.reading else None,
        "notes": s.notes,
    }


def summarize(results: List[dict]) -> str:
    def rate(num, den):
        return f"{num / den:.0%}" if den else "n/a"

    lines = [
        "| Policy | Misconceptions found (recall) | False misconception flags / session | "
        "Weak concepts found (recall) | False weak flags / session | Exactly right diagnosis | Avg questions |",
        "|---|---|---|---|---|---|---|",
    ]
    for pol in dict.fromkeys(r["policy"] for r in results):
        rs = [r for r in results if r["policy"] == pol]
        mis_tp = sum(r["mis_tp"] for r in rs); mis_fn = sum(r["mis_fn"] for r in rs)
        weak_tp = sum(r["weak_tp"] for r in rs); weak_fn = sum(r["weak_fn"] for r in rs)
        lines.append(
            f"| {pol} | {rate(mis_tp, mis_tp + mis_fn)} | {statistics.mean(r['mis_fp'] for r in rs):.2f} | "
            f"{rate(weak_tp, weak_tp + weak_fn)} | {statistics.mean(r['weak_fp'] for r in rs):.2f} | "
            f"{rate(sum(r['exact'] for r in rs), len(rs))} | {statistics.mean(r['questions'] for r in rs):.1f} |"
        )
    lines += ["", "Exactly-right diagnosis rate by student:", ""]
    pols = list(dict.fromkeys(r["policy"] for r in results))
    lines.append("| Student | " + " | ".join(pols) + " |")
    lines.append("|---|" + "---|" * len(pols))
    for p in PERSONAS:
        cells = []
        for pol in pols:
            rs = [r for r in results if r["policy"] == pol and r["student"] == p.name]
            cells.append(rate(sum(r["exact"] for r in rs), len(rs)))
        if any(c != "n/a" for c in cells):
            lines.append(f"| {p.name} | " + " | ".join(cells) + " |")
    reads = [x for r in results for x in r.get("readings", []) if x["text"] not in UNHELPFUL_REASONS]
    if reads:
        by_cause = {}
        for x in reads:
            k = "misconception" if x["cause"].startswith("misconception:") else x["cause"]
            by_cause.setdefault(k, []).append(x["match"])
        lines += ["", f"Reason reader recovered the true cause in {rate(sum(x['match'] for x in reads), len(reads))} "
                      f"of {len(reads)} informative reasons: "
                      + ", ".join(f"{k} {rate(sum(v), len(v))} (n={len(v)})" for k, v in by_cause.items())]
    agent_rs = [r for r in results if r["policy"].startswith("agent")]
    if agent_rs:
        lines += ["", f"Agent guardrail rejections: {sum(r['rejections'] for r in agent_rs)}, "
                      f"fallbacks to rule policy: {sum(r['fallbacks'] for r in agent_rs)}, "
                      f"stopped early: {sum(r['stopped_early'] for r in agent_rs)}/{len(agent_rs)} sessions"]
    return "\n".join(lines)


def print_trace(result: dict):
    print(f"\n=== {result['policy']} vs {result['student']} (run {result['run']}) ===")
    for t in result["trace"]:
        head = f"#{t['n']} [{t['policy']}] {t['action']}"
        if t["concept"]:
            head += f" → {t['concept']} L{t['level']}" + (f" (target: {t['target']})" if t["target"] else "")
        print(head)
        if t["hypothesis"]:
            print(f"   hypothesis: {t['hypothesis']}")
        print(f"   why: {t['rationale']}")
        for n in t["guardrail_notes"]:
            print(f"   guardrail: {n}")
        if t["question"]:
            res = "CORRECT" if t["correct"] else f"WRONG (picked: {t['picked_misconception'] or 'untagged'})"
            print(f"   asked {t['question']} → {res}, {t['confidence']}")
            if t.get("reason"):
                print(f'   reason: "{t["reason"]}" → read as {t["reading"]}')
            for n in t["notes"]:
                print(f"     · {n}")
    print(f"Stop: {result['stop_reason']}")
    print(f"Diagnosis: weak={result['flagged_weak']} misconceptions={result['flagged_misconceptions']} "
          f"suspected={result['suspected']} exact={result['exact']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["heuristic", "rule", "random"], choices=["agent", "agent+reasons", "heuristic", "heuristic+reasons", "rule", "random"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--run-offset", type=int, default=0,
                    help="Start run numbers here, to test on fresh seeds not used while tuning")
    ap.add_argument("--max-q", type=int, default=10)
    ap.add_argument("--min-q", type=int, default=5)
    ap.add_argument("--students", nargs="+", default=[p.name for p in PERSONAS], choices=list(PERSONA_BY_NAME))
    ap.add_argument("--show", help="Stream the full trace of one session with this student, live")
    ap.add_argument("--concepts", nargs="+", choices=list(CONCEPTS), help="Limit sessions to these concepts")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--confidence-noise", type=float, default=0.0,
                    help="Probability a simulated student's reported confidence is random (miscalibration)")
    ap.add_argument("--oracle", action="store_true",
                    help="Use a perfect reason reader (no LLM) to measure the ceiling of the reasons step")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    global CONFIDENCE_NOISE
    CONFIDENCE_NOISE = args.confidence_noise
    bank = load_bank()
    if args.show:
        for pol in args.policies:
            student = PERSONA_BY_NAME[args.show]
            print(f"\nSimulated student '{student.name}': {student.description}")
            r = run_one(pol, student, args.run_offset, args.max_q, args.min_q, bank, args.oracle,
                        stream=True, concepts=args.concepts)
            from . import live
            live.finished(r["session"])
            print(f"\nPlanted: weak={sorted(student.truth(r['session'].beliefs.concepts)['weak']) or 'none'}, "
                  f"misconceptions={list(student.misconceptions) or 'none'} → exactly right: {r['exact']}")
        from . import usage
        print("\nLLM usage and cost:\n" + usage.report())
        return

    jobs = [(pol, PERSONA_BY_NAME[s], run) for pol in args.policies for s in args.students for run in range(args.run_offset, args.run_offset + args.runs)]
    def safe_run(job):
        try:
            return run_one(job[0], job[1], job[2], args.max_q, args.min_q, bank, args.oracle)
        except Exception as e:  # one failed LLM session should not sink the whole evaluation
            print(f"FAILED {job[0]} vs {job[1].name} run {job[2]}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(safe_run, jobs) if r is not None]
    if len(results) < len(jobs):
        print(f"{len(jobs) - len(results)} of {len(jobs)} sessions failed and are excluded.")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = "-".join(args.policies) + ("-oracle" if args.oracle else "") + (
        f"-noise{args.confidence_noise:g}" if args.confidence_noise else "")
    with open(out / f"runs_{tag}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    (out / f"summary_{tag}.md").write_text(summary + "\n")
    print(summary)
    from . import usage
    print("\nLLM usage and cost:\n" + usage.report())
    print(f"\nSaved {len(results)} sessions to {out}/runs_{tag}.jsonl")


if __name__ == "__main__":
    main()
