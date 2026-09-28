# tutor/policies.py
"""
Policies decide what to ask next. All policies share the same belief state,
question bank and diagnosis, so comparing them isolates the one thing that
differs: the decision.

  - AgentPolicy (agent.py): an LLM chooses an action and justifies it,
    checked by the guardrails below.
  - RuleBasedPolicy: the original ExamCraft rule (weakest topic by accuracy,
    difficulty from accuracy thresholds).
  - RandomPolicy: random concept, random level.
"""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .bank import Question
from .beliefs import BeliefState
from .concepts import CONCEPTS, MISCONCEPTIONS, dependents

ACTIONS = {
    "screen": "First look at a concept not asked yet (usually at L2).",
    "probe_misconception": "Ask a question with a wrong option built on a specific misconception, to confirm or rule it out.",
    "step_down": "Same concept, easier level, to separate a knowledge gap from a slip or a harder-question problem.",
    "step_up": "Same concept, harder level, after a correct answer, to confirm real mastery.",
    "recheck": "Same concept, any level, with a fresh question, to tell a real gap from a one-off slip after a wrong answer.",
    "check_prerequisite": "Test a prerequisite of a concept the student is struggling with, to find the root cause.",
    "stop": "End the session because the diagnosis is clear enough and more questions would not change it.",
}


@dataclass
class Decision:
    action: str
    concept: Optional[str] = None
    level: Optional[int] = None
    target: Optional[str] = None
    rationale: str = ""
    hypothesis: str = ""
    if_correct: str = ""
    if_wrong: str = ""
    policy: str = ""
    guardrail_notes: List[str] = field(default_factory=list)


@dataclass
class Context:
    beliefs: BeliefState
    bank: Sequence[Question]
    asked_ids: set
    max_questions: int
    min_questions: int
    rng: random.Random

    @property
    def concepts(self) -> Tuple[str, ...]:
        return self.beliefs.concepts

    @property
    def answered(self) -> int:
        return len(self.beliefs.answers)

    def available(self, concept: str, target: Optional[str] = None) -> List[Question]:
        return [
            q for q in self.bank
            if q.concept == concept
            and q.id not in self.asked_ids
            and (target is None or target in q.tested_misconceptions)
        ]


def pick_question(decision: Decision, ctx: Context, informative: bool = True) -> Optional[Question]:
    """Pick an unasked question that matches the decision (nearest level if exact is gone).

    informative=True (agent) prefers questions whose wrong options test
    misconceptions not seen yet; baselines pick at random among matches.
    """
    cands = ctx.available(decision.concept, decision.target)
    if not cands:
        return None
    level = decision.level or 2
    best = min(abs(q.level - level) for q in cands)
    cands = [q for q in cands if abs(q.level - level) == best]
    if informative:
        def score(q: Question) -> int:
            untested = sum(
                1 for m in q.tested_misconceptions
                if ctx.beliefs.misconception_beliefs[m].status == "untested"
            )
            return 2 * (q.target == decision.target) + untested
        top = max(score(q) for q in cands)
        cands = [q for q in cands if score(q) == top]
    return ctx.rng.choice(sorted(cands, key=lambda q: q.id))


# ---------------- guardrails ----------------

def _allowed_concepts(ctx: Context) -> set:
    allowed = set(ctx.concepts)
    for c in ctx.concepts:
        allowed.update(CONCEPTS[c].prereqs)
    return allowed


def check_decision(d: Decision, ctx: Context) -> List[str]:
    """Return a list of rule violations (empty = the decision is allowed)."""
    if d.action not in ACTIONS:
        return [f"Unknown action '{d.action}'."]
    if d.action == "stop":
        if ctx.answered < ctx.min_questions:
            return [f"Too early to stop: {ctx.answered} answered, minimum is {ctx.min_questions}."]
        unsettled = [c for c in ctx.concepts if ctx.beliefs.concept_label(c) in ("uncertain", "not assessed")]
        if unsettled:
            return [f"Cannot stop while these concepts are still uncertain or unassessed: {', '.join(unsettled)}."]
        return []

    errors = []
    b = ctx.beliefs
    if d.concept not in CONCEPTS:
        return [f"Unknown concept '{d.concept}'."]
    if d.concept not in _allowed_concepts(ctx):
        errors.append(f"Concept '{d.concept}' is not part of this session.")
    if d.level not in (1, 2, 3):
        errors.append("Level must be 1, 2 or 3.")
        return errors

    last = b.last_answer_on(d.concept)
    if d.action == "screen" and b.concept_beliefs[d.concept].asked > 0:
        errors.append(f"'screen' is for unasked concepts; '{d.concept}' was already asked.")
    if d.action == "step_up":
        if last is None or not last.correct:
            errors.append("'step_up' requires the last answer on this concept to be correct.")
        elif d.level <= last.level:
            errors.append(f"'step_up' must go above the last level asked (L{last.level}).")
    if d.action == "step_down":
        if last is None:
            errors.append("'step_down' requires a previous question on this concept.")
        elif d.level >= last.level:
            errors.append(f"'step_down' must go below the last level asked (L{last.level}).")
    if d.action == "recheck" and not any(a.concept == d.concept and not a.correct for a in b.answers):
        errors.append("'recheck' requires a previous wrong answer on this concept.")
    if d.action == "probe_misconception":
        if d.target not in MISCONCEPTIONS:
            errors.append(f"Unknown misconception '{d.target}'.")
        elif MISCONCEPTIONS[d.target].concept != d.concept:
            errors.append(f"Misconception '{d.target}' belongs to '{MISCONCEPTIONS[d.target].concept}', not '{d.concept}'.")
        elif b.misconception_beliefs[d.target].status in ("ruled out", "confirmed"):
            errors.append(f"Misconception '{d.target}' is already {b.misconception_beliefs[d.target].status}; "
                          "probing it again wastes a question.")
    if d.action == "check_prerequisite":
        struggling = [
            c for c in dependents(d.concept)
            if any(a.concept == c and not a.correct for a in b.answers)
        ]
        if not struggling:
            errors.append(
                f"'check_prerequisite' needs a wrong answer on a concept that depends on '{d.concept}'."
            )

    if not errors:
        target = d.target if d.action == "probe_misconception" else None
        left = ctx.available(d.concept, target)
        if d.action == "step_up" and last:
            left = [q for q in left if q.level > last.level]
        if d.action == "step_down" and last:
            left = [q for q in left if q.level < last.level]
        if not left:
            errors.append(f"No unasked questions left for this move on {d.concept}" + (f" testing {target}." if target else "."))
    return errors


# ---------------- baselines ----------------

class RandomPolicy:
    name = "random"

    def decide(self, ctx: Context) -> Decision:
        concepts = [c for c in ctx.concepts if ctx.available(c)]
        concept = ctx.rng.choice(concepts)
        level = ctx.rng.choice([1, 2, 3])
        return Decision(
            action="random", concept=concept, level=level, policy=self.name,
            rationale=f"Random pick: {concept} at L{level}.",
        )


class RuleBasedPolicy:
    """Port of ExamCraft v1: weakest topic by raw accuracy, difficulty from thresholds."""
    name = "rule"

    @staticmethod
    def _difficulty(correct: int, total: int) -> int:
        if total < 3:
            return 1
        acc = correct / total
        return 1 if acc < 0.5 else (2 if acc < 0.8 else 3)

    def decide(self, ctx: Context) -> Decision:
        stats = {}
        for c in ctx.concepts:
            ans = [a for a in ctx.beliefs.answers if a.concept == c]
            stats[c] = (sum(a.correct for a in ans), len(ans))
        concepts = [c for c in ctx.concepts if ctx.available(c)]
        acc = {c: (stats[c][0] / stats[c][1] if stats[c][1] else 0.0) for c in concepts}
        lowest = min(acc.values())
        concept = ctx.rng.choice([c for c in concepts if acc[c] == lowest])
        level = self._difficulty(*stats[concept])
        return Decision(
            action="weakest_topic", concept=concept, level=level, policy=self.name,
            rationale=f"Lowest accuracy is {concept} ({acc[concept]:.0%} of {stats[concept][1]}); "
                      f"threshold rule gives L{level}.",
        )


class HeuristicPolicy:
    """The same actions as the agent, chosen by fixed if-statements instead of an LLM.

    This is the honest answer to "do I need an agent for this?": if the LLM
    cannot beat a hand-written version of its own strategy, it is not earning its place.
    """
    name = "heuristic"

    def _ok(self, d: Decision, ctx: Context) -> Optional[Decision]:
        d.policy = self.name
        return d if not check_decision(d, ctx) else None

    def decide(self, ctx: Context) -> Decision:
        b = ctx.beliefs
        last = b.answers[-1] if b.answers else None
        candidates: List[Decision] = []

        if last and not last.correct:
            m = last.attributed_misconception
            if m and last.effective_confidence != "guess" and MISCONCEPTIONS[m].concept in _allowed_concepts(ctx):
                candidates.append(Decision("probe_misconception", MISCONCEPTIONS[m].concept, last.level, m,
                                           rationale=f"#{last.number} points to '{m}'; probe it again."))
            if last.level > 1:
                candidates.append(Decision("step_down", last.concept, last.level - 1,
                                           rationale=f"#{last.number} was wrong at L{last.level}; check the easier level."))
            for p in CONCEPTS[last.concept].prereqs:
                if b.concept_beliefs[p].asked == 0:
                    candidates.append(Decision("check_prerequisite", p, 1,
                                               rationale=f"#{last.number} was wrong; check prerequisite {p}."))
        if last and last.correct and last.confidence != "sure" and last.level < 3:
            candidates.append(Decision("step_up", last.concept, last.level + 1,
                                       rationale=f"#{last.number} correct but {last.confidence}; confirm one level up."))
        for c in ctx.concepts:
            if b.concept_beliefs[c].asked == 0:
                candidates.append(Decision("screen", c, 2, rationale=f"{c} not assessed yet."))
        for m, mb in b.misconception_beliefs.items():
            if mb.status == "suspected" and MISCONCEPTIONS[m].concept in ctx.concepts:
                candidates.append(Decision("probe_misconception", MISCONCEPTIONS[m].concept, 2, m,
                                           rationale=f"'{m}' is suspected; probe it."))

        settled = all(b.concept_label(c) in ("strong", "weak", "misconception") for c in ctx.concepts)
        if settled:
            candidates.insert(0, Decision("stop", rationale="Every concept has a clear label."))

        for c in sorted(ctx.concepts, key=lambda c: -b.concept_beliefs[c].sd):
            if b.concept_label(c) == "uncertain":
                a = b.last_answer_on(c)
                if a and a.correct and a.level < 3:
                    candidates.append(Decision("step_up", c, a.level + 1, rationale=f"{c} still uncertain; go up."))
                elif a and a.level > 1:
                    candidates.append(Decision("step_down", c, a.level - 1, rationale=f"{c} still uncertain; go down."))
        candidates.append(Decision("stop", rationale="Nothing informative left to ask."))

        for d in candidates:
            ok = self._ok(d, ctx)
            if ok:
                return ok
        fallback = RuleBasedPolicy().decide(ctx)
        fallback.policy = "fallback"
        return fallback
