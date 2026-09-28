# tutor/reasons.py
"""
Reading the student's own explanation of a wrong answer.

After a wrong answer (and before the correct answer is shown), the student
writes one line on why they picked it. An LLM reads that line and says what
it reveals: a specific misconception, a knowledge gap, a slip, a guess, or
nothing clear. This is the part a rule cannot do: "I thought ridge sets
coefficients to zero" and "no idea, guessed" are both just strings.

The reading changes how the answer is scored in beliefs.py, and the raw text
is shown to the decision agent as well.
"""
from dataclasses import dataclass
from typing import Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from . import config, usage
from .bank import Question
from .concepts import CONCEPTS, MISCONCEPTIONS

KINDS = ("misconception", "gap", "slip", "guess", "unclear")


@dataclass(frozen=True)
class ReasonReading:
    kind: str                      # one of KINDS
    misconception_id: Optional[str]
    strength: str                  # "strong" or "weak"
    quote: str
    explanation: str
    text: str                      # what the student wrote


class _ReadingSchema(BaseModel):
    kind: Literal["misconception", "gap", "slip", "guess", "unclear"] = Field(
        description="misconception: the reason states a specific wrong belief. gap: the student says or shows "
                    "they don't know the concept. slip: they knew it but misread or rushed. guess: they guessed. "
                    "unclear: the reason doesn't say enough to tell.")
    misconception_id: Optional[str] = Field(
        description="If kind is misconception and it matches one in the list, its id; otherwise null.")
    strength: Literal["strong", "weak"] = Field(
        description="strong: the reason clearly states the belief or cause. weak: it only hints at it.")
    quote: str = Field(description="The words in the student's reason that support your reading (may be empty).")
    explanation: str = Field(description="One sentence on why you read it this way.")


SYSTEM_PROMPT = """You read a student's one-line explanation of a wrong multiple-choice answer and say what it reveals.
Judge only from what the student wrote, not from which option they picked: a student can pick an option for a
reason other than the one it was designed to catch.

Categories:
- slip: they say they rushed, skimmed, misread, or clicked the wrong option. Any mention of rushing or misreading
  is a slip, even if they don't say they knew the answer.
- guess: they say they guessed, had no idea, or picked at random, without describing a belief.
- gap: they say they don't know or never understood the concept (e.g. "never got what lasso does").
- misconception: they state a specific wrong belief. Match it to the list by MEANING, not wording: students
  paraphrase, and a belief stated about a specific number (e.g. "10% sensitivity means 10% false alarms")
  still matches the general misconception it instances (here, swapping sensitivity and specificity).
- unclear: vague reasons like "seemed right" or "not sure".
Only name a misconception id if the student's words express that belief."""


def _prompt(q: Question, chosen_index: int, reason: str) -> str:
    concepts = [q.concept] + list(CONCEPTS[q.concept].prereqs)
    mis = [m for c in concepts for m in CONCEPTS[c].misconceptions]
    opts = "\n".join(
        f"{'ABCD'[i]}. {o.text}{'  (correct)' if o.correct else ''}" for i, o in enumerate(q.options)
    )
    mis_list = "\n".join(f"- {m.id}: {m.description}" for m in mis)
    return (
        f"QUESTION ({CONCEPTS[q.concept].name}, level {q.level}):\n{q.stem}\n\n{opts}\n\n"
        f"The student picked {'ABCD'[chosen_index]}.\n\n"
        f"KNOWN MISCONCEPTIONS FOR THIS TOPIC:\n{mis_list}\n\n"
        f"STUDENT'S REASON: \"{reason}\""
    )


class ReasonReader:
    def __init__(self, model: Optional[str] = None, client: Optional[OpenAI] = None):
        self.model = model or config.READER_MODEL
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=8)
        return self._client

    def read(self, q: Question, chosen_index: int, reason: str) -> ReasonReading:
        from .agent import AgentError  # avoid a circular import at module load

        kwargs = {"reasoning_effort": config.AGENT_REASONING} if self.model.startswith(("gpt-5", "o")) else {}
        try:
            resp = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": _prompt(q, chosen_index, reason)}],
                response_format=_ReadingSchema,
                **kwargs,
            )
        except Exception as e:
            raise AgentError(f"Reading the reason failed: {e}") from e
        usage.record("reader", self.model, resp.usage)
        r = resp.choices[0].message.parsed
        if r is None:
            return ReasonReading("unclear", None, "weak", "", "No reading returned.", reason)
        mid = r.misconception_id if r.misconception_id in MISCONCEPTIONS else None
        kind = r.kind if not (r.kind == "misconception" and mid is None) else "unclear"
        return ReasonReading(kind, mid, r.strength, r.quote, r.explanation, reason)
