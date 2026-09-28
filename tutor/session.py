# tutor/session.py
"""One diagnostic session: decide → pick question → student answers → update beliefs."""
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .bank import Question, load_bank
from .beliefs import BeliefState, Diagnosis
from .concepts import CONCEPTS
from .policies import Context, Decision, pick_question


@dataclass
class TraceStep:
    number: int
    decision: Decision
    beliefs_before: dict
    question: Optional[Question] = None
    chosen_index: Optional[int] = None
    confidence: Optional[str] = None
    correct: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


class Session:
    def __init__(
        self,
        policy,
        concepts: Optional[Sequence[str]] = None,
        max_questions: int = 10,
        min_questions: int = 5,
        seed: Optional[int] = None,
        bank: Optional[Sequence[Question]] = None,
    ):
        self.policy = policy
        self.beliefs = BeliefState(concepts=tuple(concepts or CONCEPTS.keys()))
        self.ctx = Context(
            beliefs=self.beliefs,
            bank=bank if bank is not None else load_bank(),
            asked_ids=set(),
            max_questions=max_questions,
            min_questions=min(min_questions, max_questions),
            rng=random.Random(seed),
        )
        self.trace: List[TraceStep] = []
        self.finished = False
        self.stop_reason = ""

    @property
    def current(self) -> Optional[TraceStep]:
        if self.trace and self.trace[-1].chosen_index is None and self.trace[-1].question:
            return self.trace[-1]
        return None

    def next_question(self) -> Optional[TraceStep]:
        """Ask the policy for the next move. Returns None when the session ends."""
        if self.finished:
            return None
        if self.current:
            return self.current
        if self.ctx.answered >= self.ctx.max_questions:
            self._finish(f"Reached the {self.ctx.max_questions}-question budget.")
            return None

        before = self.beliefs.snapshot()
        decision = self.policy.decide(self.ctx)
        step = TraceStep(number=self.ctx.answered + 1, decision=decision, beliefs_before=before)
        self.trace.append(step)

        if decision.action == "stop":
            self._finish(f"The agent chose to stop: {decision.rationale}")
            return None

        informative = decision.policy in ("agent", "heuristic")
        q = pick_question(decision, self.ctx, informative=informative)
        if q is None:
            # Only baselines/fallback reach here (guardrails check availability for the agent).
            for c in self.ctx.concepts:
                if self.ctx.available(c):
                    decision.guardrail_notes.append(f"No question left for {decision.concept}; switched to {c}.")
                    decision.concept = c
                    q = pick_question(decision, self.ctx, informative=informative)
                    break
        if q is None:
            self._finish("The question bank ran out for these concepts.")
            return None

        step.question = q
        self.ctx.asked_ids.add(q.id)
        return step

    def submit(self, chosen_index: int, confidence: str) -> TraceStep:
        step = self.current
        if step is None:
            raise RuntimeError("No question is waiting for an answer.")
        step.chosen_index = chosen_index
        step.confidence = confidence
        step.correct = step.question.options[chosen_index].correct
        step.notes = self.beliefs.update(step.question, chosen_index, confidence)
        return step

    def diagnosis(self) -> Diagnosis:
        return self.beliefs.diagnose()

    def _finish(self, reason: str):
        self.finished = True
        self.stop_reason = reason
