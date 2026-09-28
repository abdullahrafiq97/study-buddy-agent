# tutor/agent.py
"""
The agentic part: an LLM reads the belief state and the answer history,
forms a hypothesis about the student, and picks the next move from a fixed
set of actions. Every choice is checked by the guardrails in policies.py;
a rejected choice is sent back once with the reason, and if the second
attempt also fails the session falls back to the rule-based policy (and
says so in the trace).
"""
from typing import List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from . import config
from .concepts import CONCEPTS, LEVELS
from .policies import ACTIONS, Context, Decision, RuleBasedPolicy, check_decision


class AgentError(RuntimeError):
    pass


class AgentChoice(BaseModel):
    hypothesis: str = Field(description="Your current read of the student in 1-2 sentences, citing answer numbers.")
    action: Literal["screen", "probe_misconception", "step_down", "step_up", "recheck", "check_prerequisite", "stop"]
    concept_id: Optional[str] = Field(description="Concept id for the next question (null only when stopping).")
    level: Optional[Literal[1, 2, 3]] = Field(description="Difficulty level 1-3 (null only when stopping).")
    target_misconception: Optional[str] = Field(description="Misconception id, required for probe_misconception, else null.")
    rationale: str = Field(description="Why this move beats the alternatives, citing evidence.")
    if_correct: str = Field(description="What a correct answer would tell you.")
    if_wrong: str = Field(description="What a wrong answer would tell you (mention which wrong option matters).")


SYSTEM_PROMPT = f"""You are the decision-maker inside a diagnostic quiz on regression and classification.
Your goal: in as few questions as possible, work out for each concept in the session whether the student
(a) has mastered it, (b) has a knowledge gap, or (c) holds a specific misconception, and whether a weak
concept is really caused by a weak prerequisite.

You do not write questions. A question bank already exists. Every question has one correct option and
wrong options, some tagged with the misconception a student would need to hold to pick them. After each
answer the student also reports confidence: sure, unsure, or guess. The numbers in the state (mastery
mean ± sd, misconception statuses) are computed for you; do not recompute them, reason about them.

Difficulty levels:
{chr(10).join(f"- {v}" for v in LEVELS.values())}

Actions:
{chr(10).join(f"- {k}: {v}" for k, v in ACTIONS.items())}

Hard rules (a guardrail rejects violations):
- screen only for concepts with 0 questions asked.
- step_up only right after a correct answer on that concept, to a higher level than last asked.
- step_down only to a lower level than last asked on that concept.
- recheck only on a concept with a previous wrong answer.
- probe_misconception needs target_misconception belonging to concept_id, not already 'ruled out' or 'confirmed'.
- check_prerequisite: concept_id must be a prerequisite of a concept the student has answered wrong.
- stop only after the minimum number of questions, and only when no session concept is 'uncertain' or 'not assessed'.

How to read evidence:
- Wrong + picked a tagged option + sure/unsure: that misconception is suspected. One hit is not proof;
  probe it again with a different question before concluding. Two confident hits confirm it.
- Wrong while GUESSING counts as a gap signal, even if the guessed option happened to be tagged. A guess
  that lands on a tagged option is chance, not a misconception lead: do not spend a probe on it.
- Wrong + untagged option: points to a gap. Step down to see whether the basics hold; if the basics fail
  and the concept has prerequisites, check the prerequisite.
- Correct but guess/unsure: weak evidence of mastery; consider stepping up or re-checking.
- Correct and sure at L2/L3: good sign, but one answer is not mastery (the label stays 'uncertain' until there
  is enough evidence). A correct answer only rules out the misconceptions that question offered.
- Student's reason (when given): after a wrong answer the student explains their choice in their own words,
  and the history shows it with a reading. The reading already updated the numbers, but read the words
  yourself too: a reason can reveal a misconception the picked option was not tagged with, show a gap
  ("never understood lasso"), or show a slip ("misread the question"). A slip is not evidence of weakness.
- A confirmed misconception explains the wrong answers it caused, but not others. Concept labels use only
  the wrong answers a confirmed misconception does not explain, so a concept can be both 'weak' and have
  a misconception. After confirming one, check whether the student also has a gap in that concept.

Budget priorities, in order:
1. Screen every session concept once.
2. Resolve concepts with unexplained wrong answers (the "gap signals" column): step down, recheck, or check
   a prerequisite until the concept is labelled weak or the miss is shown to be a slip. A real gap matters
   more to the student than an unconfirmed misconception.
3. Probe misconceptions that have a confident hit (hits >= 0.7).
4. Firm up concepts that are still 'uncertain' after correct answers (step up).
Do not spend more than 3 questions in a row on one concept unless it is the only unresolved one.
- Stop when every session concept is labelled strong, weak, or misconception.

Always cite answer numbers (e.g., "#3") when explaining."""


def gap_signals(b, concept_id: str) -> int:
    """Wrong answers on a concept that no confident misconception explains (slips excluded)."""
    return sum(
        1 for a in b.answers
        if a.concept == concept_id and not a.correct and a.reading_kind != "slip"
        and not (a.attributed_misconception and a.effective_confidence != "guess")
    )


def render_state(ctx: Context) -> str:
    b = ctx.beliefs
    n = ctx.answered
    lines = [
        f"SESSION: next is question #{n + 1}; at most {ctx.max_questions} questions; "
        f"you may stop after {ctx.min_questions} (answered so far: {n}).",
        "",
        "CONCEPTS (id | name | prerequisites | mastery mean ± sd | asked | label | gap signals | questions left L1/L2/L3):",
    ]
    session_and_prereqs = list(ctx.concepts) + [
        p for c in ctx.concepts for p in CONCEPTS[c].prereqs if p not in ctx.concepts
    ]
    for cid in dict.fromkeys(session_and_prereqs):
        c = CONCEPTS[cid]
        cb = b.concept_beliefs[cid]
        left = [sum(1 for q in ctx.available(cid) if q.level == lv) for lv in (1, 2, 3)]
        tag = "" if cid in ctx.concepts else " [prerequisite only, not in session]"
        lines.append(
            f"- {cid} | {c.name} | prereqs: {', '.join(c.prereqs) or 'none'} | "
            f"{cb.mean:.2f} ± {cb.sd:.2f} | {cb.asked} | {b.concept_label(cid)} | "
            f"{gap_signals(b, cid)} | {left[0]}/{left[1]}/{left[2]}{tag}"
        )
    lines += ["", "MISCONCEPTIONS (id [status, hits, avoided]: description):"]
    for cid in ctx.concepts:
        for m in CONCEPTS[cid].misconceptions:
            mb = b.misconception_beliefs[m.id]
            lines.append(
                f"- {cid} / {m.id} [{mb.status}, hits={mb.hits:.1f}, avoided={mb.avoided:.1f}]: {m.description}"
            )
    lines += ["", "ANSWER HISTORY (oldest first):"]
    if not b.answers:
        lines.append("- none yet")
    for a in b.answers:
        picked = (
            "correct option" if a.correct
            else f"wrong option tagged '{a.chosen_misconception}'" if a.chosen_misconception
            else "wrong option with no misconception tag"
        )
        offered = ", ".join(a.offered) or "none"
        line = (
            f"- #{a.number} {a.concept} L{a.level}: {'CORRECT' if a.correct else 'WRONG'} "
            f"({picked}), confidence={a.confidence}. Misconceptions offered: {offered}"
        )
        if a.reason is not None:
            line += (f"\n    Student's reason: \"{a.reason}\" → read as {a.reading_kind}"
                     + (f" ({a.attributed_misconception})" if a.reading_kind == "misconception" else ""))
        lines.append(line)
    return "\n".join(lines)


class AgentPolicy:
    name = "agent"

    def __init__(self, model: Optional[str] = None, reasoning: Optional[str] = None, client: Optional[OpenAI] = None):
        self.model = model or config.AGENT_MODEL
        self.reasoning = reasoning or config.AGENT_REASONING
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not config.OPENAI_API_KEY:
                raise AgentError("OPENAI_API_KEY is not set in .env")
            self._client = OpenAI(api_key=config.OPENAI_API_KEY, max_retries=8)  # backs off on rate limits
        return self._client

    def _ask(self, messages: List[dict]) -> AgentChoice:
        kwargs = {}
        if self.model.startswith(("gpt-5", "o")):
            kwargs["reasoning_effort"] = self.reasoning
        try:
            resp = self.client.chat.completions.parse(
                model=self.model, messages=messages, response_format=AgentChoice, **kwargs
            )
        except Exception as e:  # surface API problems (quota, auth) clearly
            raise AgentError(f"LLM call failed: {e}") from e
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise AgentError("LLM returned no structured decision (refusal or empty).")
        return parsed

    def decide(self, ctx: Context) -> Decision:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_state(ctx)},
        ]
        notes: List[str] = []
        for _ in range(2):
            choice = self._ask(messages)
            d = Decision(
                action=choice.action,
                concept=choice.concept_id,
                level=choice.level,
                target=choice.target_misconception if choice.action == "probe_misconception" else None,
                rationale=choice.rationale,
                hypothesis=choice.hypothesis,
                if_correct=choice.if_correct,
                if_wrong=choice.if_wrong,
                policy=self.name,
            )
            errors = check_decision(d, ctx)
            if not errors:
                d.guardrail_notes = notes
                return d
            notes.append(f"Rejected {d.action}({d.concept}, L{d.level}): " + " ".join(errors))
            messages.append({"role": "assistant", "content": choice.model_dump_json()})
            messages.append({
                "role": "user",
                "content": "Guardrail rejected that move: " + " ".join(errors) + " Choose a different, valid move.",
            })

        fallback = RuleBasedPolicy().decide(ctx)
        fallback.policy = "fallback"
        fallback.guardrail_notes = notes + ["Two invalid moves in a row; fell back to the rule-based policy."]
        return fallback

