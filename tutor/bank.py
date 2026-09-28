# tutor/bank.py
"""Load and validate the question bank."""
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .concepts import CONCEPTS, MISCONCEPTIONS

BANK_PATH = Path(__file__).parent / "data" / "question_bank.json"


@dataclass(frozen=True)
class Option:
    text: str
    correct: bool
    misconception: Optional[str]


@dataclass(frozen=True)
class Question:
    id: str
    concept: str
    level: int
    target: str
    stem: str
    options: Tuple[Option, ...]
    explanation: str
    source: str

    @property
    def correct_index(self) -> int:
        return next(i for i, o in enumerate(self.options) if o.correct)

    @property
    def tested_misconceptions(self) -> Tuple[str, ...]:
        """Misconceptions that have a tagged wrong option in this question."""
        return tuple(dict.fromkeys(o.misconception for o in self.options if o.misconception))


def _validate(raw: dict) -> List[str]:
    errors = []
    qid = raw.get("id", "?")
    if raw.get("concept") not in CONCEPTS:
        errors.append(f"{qid}: unknown concept {raw.get('concept')}")
    if raw.get("level") not in (1, 2, 3):
        errors.append(f"{qid}: level must be 1, 2 or 3")
    opts = raw.get("options", [])
    if len(opts) != 4:
        errors.append(f"{qid}: needs exactly 4 options")
    if sum(1 for o in opts if o.get("correct")) != 1:
        errors.append(f"{qid}: needs exactly one correct option")
    for o in opts:
        m = o.get("misconception")
        if m and m not in MISCONCEPTIONS:
            errors.append(f"{qid}: unknown misconception {m}")
        if m and o.get("correct"):
            errors.append(f"{qid}: correct option cannot carry a misconception")
    target = raw.get("target")
    if target not in MISCONCEPTIONS:
        errors.append(f"{qid}: unknown target {target}")
    elif MISCONCEPTIONS[target].concept != raw.get("concept"):
        errors.append(f"{qid}: target {target} belongs to another concept")
    elif target not in [o.get("misconception") for o in opts]:
        errors.append(f"{qid}: no option is tagged with target {target}")
    return errors


def load_bank(path: Path = BANK_PATH) -> List[Question]:
    raw_bank = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = [e for raw in raw_bank for e in _validate(raw)]
    ids = [raw["id"] for raw in raw_bank]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")
    if errors:
        raise ValueError("Invalid question bank:\n" + "\n".join(errors))

    questions = []
    for i, raw in enumerate(raw_bank):
        opts = [
            Option(text=o["text"], correct=bool(o.get("correct")), misconception=o.get("misconception"))
            for o in raw["options"]
        ]
        # The bank lists the correct answer first. Shuffle the wrong options per question,
        # then place the correct one so each position (A-D) is correct equally often.
        correct = next(o for o in opts if o.correct)
        wrong = [o for o in opts if not o.correct]
        rng = random.Random(raw["id"])
        rng.shuffle(wrong)
        wrong.insert(i % 4, correct)
        opts = wrong
        questions.append(
            Question(
                id=raw["id"],
                concept=raw["concept"],
                level=raw["level"],
                target=raw["target"],
                stem=raw["stem"],
                options=tuple(opts),
                explanation=raw["explanation"],
                source=raw["source"],
            )
        )
    return questions
