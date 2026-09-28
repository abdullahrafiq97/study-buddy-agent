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

    def answer(self, q: Question, rng: random.Random) -> Tuple[int, str]:
        held = [i for i, o in enumerate(q.options) if o.misconception in self.misconceptions]
        if held and rng.random() < FALLS_FOR_HELD:
            return rng.choice(held), ("sure" if rng.random() < 0.7 else "unsure")
        wrong = [i for i, o in enumerate(q.options) if not o.correct]
        if rng.random() < self.effective_mastery(q.concept):
            if rng.random() < KNOWS_CORRECT[q.level]:
                conf = "sure" if q.level < 3 or rng.random() < 0.6 else "unsure"
                return q.correct_index, conf
            return rng.choice(wrong), "unsure"  # a slip
        return rng.randrange(4), ("guess" if rng.random() < 0.7 else "unsure")

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


def make_policy(name: str):
    if name == "agent":
        from .agent import AgentPolicy
        return AgentPolicy()
    return {"rule": RuleBasedPolicy, "random": RandomPolicy, "heuristic": HeuristicPolicy}[name]()


def run_one(policy_name: str, student: SimStudent, run: int, max_q: int, min_q: int, bank) -> dict:
    session = Session(make_policy(policy_name), max_questions=max_q, min_questions=min_q,
                      seed=1000 + run, bank=bank)
    rng = random.Random(f"{student.name}-{run}")
    while True:
        step = session.next_question()
        if step is None:
            break
        idx, conf = student.answer(step.question, rng)
        session.submit(idx, conf)

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
        "trace": [trace_to_dict(s) for s in session.trace],
    }
    result["exact"] = all(result[k] == 0 for k in ("mis_fp", "mis_fn", "weak_fp", "weak_fn"))
    return result


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
    agent_rs = [r for r in results if r["policy"] == "agent"]
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
            for n in t["notes"]:
                print(f"     · {n}")
    print(f"Stop: {result['stop_reason']}")
    print(f"Diagnosis: weak={result['flagged_weak']} misconceptions={result['flagged_misconceptions']} "
          f"suspected={result['suspected']} exact={result['exact']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["heuristic", "rule", "random"], choices=["agent", "heuristic", "rule", "random"])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-q", type=int, default=10)
    ap.add_argument("--min-q", type=int, default=5)
    ap.add_argument("--students", nargs="+", default=[p.name for p in PERSONAS], choices=list(PERSONA_BY_NAME))
    ap.add_argument("--show", help="Print the full trace of run 0 for this student")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    bank = load_bank()
    if args.show:
        for pol in args.policies:
            print_trace(run_one(pol, PERSONA_BY_NAME[args.show], 0, args.max_q, args.min_q, bank))
        return

    jobs = [(pol, PERSONA_BY_NAME[s], run) for pol in args.policies for s in args.students for run in range(args.runs)]
    def safe_run(job):
        try:
            return run_one(job[0], job[1], job[2], args.max_q, args.min_q, bank)
        except Exception as e:  # one failed LLM session should not sink the whole evaluation
            print(f"FAILED {job[0]} vs {job[1].name} run {job[2]}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [r for r in ex.map(safe_run, jobs) if r is not None]
    if len(results) < len(jobs):
        print(f"{len(jobs) - len(results)} of {len(jobs)} sessions failed and are excluded.")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = "-".join(args.policies)
    with open(out / f"runs_{tag}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    (out / f"summary_{tag}.md").write_text(summary + "\n")
    print(summary)
    print(f"\nSaved {len(results)} sessions to {out}/runs_{tag}.jsonl")


if __name__ == "__main__":
    main()
