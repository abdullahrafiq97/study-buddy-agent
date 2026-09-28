# tutor/beliefs.py
"""
What the agent believes about the student, updated deterministically after
every answer. The LLM never edits these numbers; it reads them and decides
what to ask next.

Mastery per concept is a Beta(alpha, beta) distribution, starting at
Beta(1, 1) ("no idea"). Each answer adds evidence, weighted by:
  - difficulty: missing an easy question says more than missing a hard one,
    and getting a hard one right says more than getting an easy one right;
  - confidence: a correct answer marked "guess" barely counts.

Misconceptions are tracked separately: picking a wrong option tagged with
misconception M is a "hit" for M; seeing M's option and not picking it is
evidence against M.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .bank import Question
from .concepts import CONCEPTS, MISCONCEPTIONS

CONFIDENCES = ("sure", "unsure", "guess")

CORRECT_WEIGHT = {1: 0.5, 2: 1.0, 3: 1.5}
WRONG_WEIGHT = {1: 1.5, 2: 1.0, 3: 0.5}
CORRECT_CONF = {"sure": 1.0, "unsure": 0.6, "guess": 0.2}

HIT_CONF = {"sure": 1.0, "unsure": 0.7, "guess": 0.3}
AVOID_CORRECT_CONF = {"sure": 1.0, "unsure": 0.6, "guess": 0.2}
AVOID_WRONG_OTHER = 0.3  # wrong, but picked a different option than M's

# Weights from the student's written reason (see reasons.py).
READING_HIT = {"strong": 1.0, "weak": 0.6}
SLIP_FACTOR = 0.3  # a wrong answer the reason shows was a slip counts 30% as much

# Thresholds used for labels in the diagnosis.
MIN_EVIDENCE = 2.0
STRONG_AT = 0.65
WEAK_AT = 0.40
CONFIRM_HITS = 1.5
RULE_OUT_AVOIDS = 1.5


@dataclass
class Answer:
    number: int
    question_id: str
    concept: str
    level: int
    correct: bool
    confidence: str
    chosen_misconception: Optional[str]      # tag on the option they picked
    offered: Tuple[str, ...]
    attributed_misconception: Optional[str] = None  # what the evidence says caused it (tag or reason)
    effective_confidence: str = ""                  # confidence after reading the reason
    wrong_weight: float = 0.0                       # how much this wrong answer counts against mastery
    reason: Optional[str] = None
    reading_kind: Optional[str] = None


@dataclass
class ConceptBelief:
    alpha: float = 1.0
    beta: float = 1.0
    asked: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sd(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    @property
    def evidence(self) -> float:
        return self.alpha + self.beta - 2.0


@dataclass
class MisconceptionBelief:
    hits: float = 0.0
    avoided: float = 0.0
    exposures: int = 0

    @property
    def status(self) -> str:
        if self.exposures == 0:
            return "untested"
        if self.hits >= CONFIRM_HITS and self.hits > self.avoided:
            return "confirmed"
        if self.hits > 0:
            return "suspected" if self.hits >= self.avoided else "unlikely"
        if self.avoided >= RULE_OUT_AVOIDS:
            return "ruled out"
        return "unlikely"


@dataclass
class Diagnosis:
    labels: Dict[str, str]
    confirmed: List[str]
    suspected: List[str]
    root_causes: List[Tuple[str, str]]  # (weak concept, weak prerequisite)

    @property
    def weak(self) -> List[str]:
        return [c for c, lab in self.labels.items() if lab == "weak"]


@dataclass
class BeliefState:
    concepts: Tuple[str, ...]
    concept_beliefs: Dict[str, ConceptBelief] = field(default_factory=dict)
    misconception_beliefs: Dict[str, MisconceptionBelief] = field(default_factory=dict)
    answers: List[Answer] = field(default_factory=list)

    def __post_init__(self):
        for cid in CONCEPTS:
            self.concept_beliefs.setdefault(cid, ConceptBelief())
        for mid in MISCONCEPTIONS:
            self.misconception_beliefs.setdefault(mid, MisconceptionBelief())

    # ---------- updating ----------

    def update(self, q: Question, chosen_index: int, confidence: str, reading=None) -> List[str]:
        """Apply one answer. Returns human-readable notes on what changed.

        reading: optional ReasonReading of the student's explanation of a wrong
        answer. It can re-attribute the answer to a different misconception,
        mark it a guess or gap (no misconception evidence), or a slip (counts less).
        """
        if confidence not in CONFIDENCES:
            raise ValueError(f"confidence must be one of {CONFIDENCES}")
        chosen = q.options[chosen_index]
        correct = chosen.correct
        cb = self.concept_beliefs[q.concept]
        before = cb.mean

        attributed = None if correct else chosen.misconception
        effective_conf = confidence
        wrong_weight = 0.0 if correct else WRONG_WEIGHT[q.level]
        hit_weight = HIT_CONF[confidence]
        notes = []
        if reading is not None and not correct:
            if reading.kind == "misconception":
                attributed = reading.misconception_id
                hit_weight = READING_HIT[reading.strength]
            elif reading.kind in ("guess", "gap", "slip"):
                attributed = None
                if reading.kind == "guess":
                    effective_conf = "guess"
                if reading.kind == "slip":
                    wrong_weight *= SLIP_FACTOR
            notes.append(f"Reason read as: {reading.kind}"
                         + (f" ({reading.misconception_id}, {reading.strength})" if reading.misconception_id else ""))

        if correct:
            cb.alpha += CORRECT_WEIGHT[q.level] * CORRECT_CONF[confidence]
        else:
            cb.beta += wrong_weight
        cb.asked += 1

        notes.insert(0,
            f"{CONCEPTS[q.concept].name}: mastery {before:.2f} → {cb.mean:.2f} "
            f"(±{cb.sd:.2f}; L{q.level} {'correct' if correct else 'wrong'}, {effective_conf})"
        )

        touched = list(q.tested_misconceptions) + ([attributed] if attributed and attributed not in q.tested_misconceptions else [])
        for mid in touched:
            mb = self.misconception_beliefs[mid]
            old_status = mb.status
            mb.exposures += 1
            if mid == attributed:
                mb.hits += hit_weight
            elif correct:
                mb.avoided += AVOID_CORRECT_CONF[confidence]
            elif mid == chosen.misconception:
                pass  # picked its option, but the reason says it wasn't that belief: no evidence either way
            else:
                mb.avoided += AVOID_WRONG_OTHER
            if mb.status != old_status:
                notes.append(f"Misconception '{mid}': {old_status} → {mb.status}")

        self.answers.append(
            Answer(
                number=len(self.answers) + 1,
                question_id=q.id,
                concept=q.concept,
                level=q.level,
                correct=correct,
                confidence=confidence,
                chosen_misconception=chosen.misconception,
                offered=q.tested_misconceptions,
                attributed_misconception=attributed,
                effective_confidence=effective_conf,
                wrong_weight=wrong_weight,
                reason=reading.text if reading is not None else None,
                reading_kind=reading.kind if reading is not None else None,
            )
        )
        return notes

    # ---------- reading ----------

    def last_answer_on(self, concept_id: str) -> Optional[Answer]:
        for a in reversed(self.answers):
            if a.concept == concept_id:
                return a
        return None

    def gap_belief(self, concept_id: str) -> ConceptBelief:
        """Mastery from answers NOT explained by a confirmed misconception.

        A student who keeps picking the same misconception's option is not
        necessarily weak on the whole concept; this separates the two.
        """
        gb = ConceptBelief()
        for a in self.answers:
            if a.concept != concept_id:
                continue
            gb.asked += 1
            if a.correct:
                gb.alpha += CORRECT_WEIGHT[a.level] * CORRECT_CONF[a.confidence]
            elif not (a.attributed_misconception
                      and self.misconception_beliefs[a.attributed_misconception].status == "confirmed"):
                gb.beta += a.wrong_weight
        return gb

    def concept_label(self, concept_id: str) -> str:
        if self.concept_beliefs[concept_id].asked == 0:
            return "not assessed"
        gb = self.gap_belief(concept_id)
        if gb.evidence >= MIN_EVIDENCE and gb.mean <= WEAK_AT:
            return "weak"
        if any(
            self.misconception_beliefs[m.id].status == "confirmed"
            for m in CONCEPTS[concept_id].misconceptions
        ):
            return "misconception"
        if gb.evidence >= MIN_EVIDENCE and gb.mean >= STRONG_AT:
            return "strong"
        return "uncertain"

    def diagnose(self) -> Diagnosis:
        labels = {c: self.concept_label(c) for c in self.concepts}
        confirmed = [m for m, b in self.misconception_beliefs.items() if b.status == "confirmed"]
        suspected = [m for m, b in self.misconception_beliefs.items() if b.status == "suspected"]
        root_causes = [
            (c, p)
            for c in self.concepts
            if labels[c] == "weak"
            for p in CONCEPTS[c].prereqs
            if self.concept_label(p) == "weak"
        ]
        return Diagnosis(labels=labels, confirmed=confirmed, suspected=suspected, root_causes=root_causes)

    def snapshot(self) -> dict:
        """Compact copy of the numbers, stored with each trace step."""
        return {
            "concepts": {
                c: {
                    "mean": round(self.concept_beliefs[c].mean, 3),
                    "sd": round(self.concept_beliefs[c].sd, 3),
                    "asked": self.concept_beliefs[c].asked,
                    "label": self.concept_label(c),
                }
                for c in self.concepts
            },
            "misconceptions": {
                m: self.misconception_beliefs[m].status
                for m in MISCONCEPTIONS
                if self.misconception_beliefs[m].exposures > 0
            },
        }
