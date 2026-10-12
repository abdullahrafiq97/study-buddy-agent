# tutor/render.py
"""HTML / Markdown rendering for the app. No logic lives here."""
import html
from typing import List, Optional

from .beliefs import BeliefState
from .concepts import CONCEPTS, MISCONCEPTIONS
from .session import Session, TraceStep

LETTERS = "ABCD"

ACTION_LABELS = {
    "screen": "Screen a new concept",
    "probe_misconception": "Probe a suspected misconception",
    "step_down": "Step down a level",
    "step_up": "Step up a level",
    "recheck": "Recheck with a fresh question",
    "check_prerequisite": "Check a prerequisite",
    "stop": "Stop: diagnosis is clear",
    "weakest_topic": "Weakest topic (fixed rule)",
    "random": "Random pick",
}

LABEL_CLASS = {
    "strong": "ok", "weak": "bad", "misconception": "warn",
    "uncertain": "muted", "not assessed": "muted",
}

CSS = """
.sb-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px 14px;
           background: var(--block-background-fill); margin-bottom: 10px; }
.sb-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
.sb-kicker { font-size: 0.75rem; text-transform: uppercase; letter-spacing: .04em;
             color: var(--body-text-color-subdued); margin-bottom: 4px; }
.sb-row { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) auto; gap: 8px;
          align-items: center; margin: 5px 0; font-size: 0.88rem; }
.sb-bar { height: 8px; border-radius: 4px; background: var(--border-color-primary); position: relative; overflow: hidden; }
.sb-bar > span { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 4px; background: var(--color-accent); }
.sb-bar > i { position: absolute; top: -2px; bottom: -2px; background: var(--body-text-color-subdued); opacity: .25; }
.sb-pill { font-size: 0.72rem; padding: 1px 8px; border-radius: 999px; white-space: nowrap;
           border: 1px solid currentColor; }
.sb-pill.ok { color: #1a7f37; } .sb-pill.bad { color: #cf222e; } .sb-pill.warn { color: #9a6700; }
.sb-pill.muted { color: var(--body-text-color-subdued); }
@media (prefers-color-scheme: dark) {
  .sb-pill.ok { color: #3fb950; } .sb-pill.bad { color: #f85149; } .sb-pill.warn { color: #d29922; }
}
.dark .sb-pill.ok { color: #3fb950; } .dark .sb-pill.bad { color: #f85149; } .dark .sb-pill.warn { color: #d29922; }
.sb-note { font-size: 0.85rem; color: var(--body-text-color-subdued); margin: 4px 0 0 0; }
.sb-guard { font-size: 0.82rem; border-left: 3px solid #d29922; padding-left: 8px; margin-top: 8px; }
.sb-mis { font-size: 0.85rem; margin: 4px 0; }
"""


def esc(s: Optional[str]) -> str:
    return html.escape(s or "")


def option_choices(step: TraceStep) -> List[str]:
    return [f"{LETTERS[i]}. {o.text}" for i, o in enumerate(step.question.options)]


def question_md(step: TraceStep, max_q: int) -> str:
    q = step.question
    c = CONCEPTS[q.concept]
    return (
        f"#### Question {step.number} of up to {max_q}\n"
        f"**{c.name}** · Level {q.level} · _{q.source}_\n\n"
        f"{q.stem}"
    )


def decision_html(step: Optional[TraceStep]) -> str:
    if step is None:
        return "<div class='sb-card'><div class='sb-kicker'>Why this question</div>Start a session to see the reasoning.</div>"
    d = step.decision
    title = ACTION_LABELS.get(d.action, d.action)
    target = ""
    if d.concept:
        target = f"{esc(CONCEPTS[d.concept].name)} · L{d.level}"
        if d.target:
            target += f" · testing <code>{esc(d.target)}</code>"
    parts = [
        "<div class='sb-card'>",
        f"<div class='sb-kicker'>Decision #{step.number} · by {esc(d.policy)}</div>",
        f"<h4>{esc(title)}</h4>",
        f"<div>{target}</div>" if target else "",
    ]
    if d.hypothesis:
        parts.append(f"<p><b>Current read:</b> {esc(d.hypothesis)}</p>")
    parts.append(f"<p><b>Why:</b> {esc(d.rationale)}</p>")
    if d.if_correct or d.if_wrong:
        parts.append(f"<p class='sb-note'>If correct → {esc(d.if_correct)}<br>If wrong → {esc(d.if_wrong)}</p>")
    for n in d.guardrail_notes:
        parts.append(f"<div class='sb-guard'>Guardrail: {esc(n)}</div>")
    parts.append("</div>")
    return "".join(parts)


def beliefs_html(b: BeliefState) -> str:
    rows = []
    for cid in b.concepts:
        cb = b.concept_beliefs[cid]
        label = b.concept_label(cid)
        lo = max(0.0, cb.mean - cb.sd) * 100
        hi = min(1.0, cb.mean + cb.sd) * 100
        rows.append(
            "<div class='sb-row'>"
            f"<div>{esc(CONCEPTS[cid].name)} <span class='sb-note'>({cb.asked} asked)</span></div>"
            f"<div class='sb-bar' title='mastery {cb.mean:.2f} ± {cb.sd:.2f}'>"
            f"<i style='left:{lo:.0f}%;width:{hi - lo:.0f}%'></i><span style='width:{cb.mean * 100:.0f}%'></span></div>"
            f"<span class='sb-pill {LABEL_CLASS[label]}'>{esc(label)}</span>"
            "</div>"
        )
    mis = []
    for mid, mb in b.misconception_beliefs.items():
        if mb.exposures == 0:
            continue
        cls = {"confirmed": "bad", "suspected": "warn"}.get(mb.status, "muted")
        mis.append(
            f"<div class='sb-mis'><span class='sb-pill {cls}'>{esc(mb.status)}</span> "
            f"<code>{esc(mid)}</code> <span class='sb-note'>hits {mb.hits:.1f} · avoided {mb.avoided:.1f}</span></div>"
        )
    return (
        "<div class='sb-card'><div class='sb-kicker'>What the agent believes (mastery ± uncertainty)</div>"
        + "".join(rows) + "</div>"
        + "<div class='sb-card'><div class='sb-kicker'>Misconceptions tested so far</div>"
        + ("".join(mis) or "<span class='sb-note'>None tested yet.</span>") + "</div>"
    )


def feedback_md(step: TraceStep) -> str:
    q = step.question
    chosen = q.options[step.chosen_index]
    right = LETTERS[q.correct_index]
    if step.correct:
        head = f"✅ **Correct** ({right})."
    else:
        head = f"❌ **Not quite.** You picked {LETTERS[step.chosen_index]}; the answer is {right}."
        if chosen.misconception:
            head += f"\n\nThat option is what you'd pick if you believe: _{MISCONCEPTIONS[chosen.misconception].description}_"
    if step.reading is not None:
        r = step.reading
        read = {"misconception": f"a specific belief: _{MISCONCEPTIONS[r.misconception_id].description}_"
                                 if r.misconception_id else "a specific belief",
                "gap": "a gap in what you know about this topic",
                "slip": "a slip: you likely knew it but misread or rushed",
                "guess": "a guess", "unclear": "not enough to tell"}[r.kind]
        head += f"\n\n**How I read your reason** (\"{r.text}\"): {read}. {r.explanation}"
    notes = "\n".join(f"- {n}" for n in step.notes)
    return f"{head}\n\n**Explanation:** {q.explanation}\n\n**Belief update:**\n{notes}"


def trace_md(session: Session) -> str:
    if not session.trace:
        return "_No steps yet._"
    out = []
    for s in session.trace:
        d = s.decision
        line = f"**#{s.number} {ACTION_LABELS.get(d.action, d.action)}**"
        if d.concept:
            line += f" → {d.concept} L{d.level}" + (f" (testing `{d.target}`)" if d.target else "")
        out.append(line)
        if d.hypothesis:
            out.append(f"- read: {d.hypothesis}")
        out.append(f"- why: {d.rationale}")
        out += [f"- guardrail: {n}" for n in d.guardrail_notes]
        if s.chosen_index is not None:
            res = "correct" if s.correct else "wrong"
            m = s.question.options[s.chosen_index].misconception
            out.append(f"- asked `{s.question.id}` → {res}, {s.confidence}" + (f", picked `{m}`" if m else ""))
            if s.reading is not None:
                out.append(f'  - reason: "{s.reading.text}" → {s.reading.kind}')
            out += [f"  - {n}" for n in s.notes]
        out.append("")
    return "\n".join(out)


def diagnosis_md(session: Session) -> str:
    b = session.beliefs
    diag = session.diagnosis()
    out = [
        f"## Diagnosis after {len(b.answers)} questions",
        f"_{session.stop_reason}_",
        "",
        "| Concept | Verdict | Mastery | Questions |",
        "|---|---|---|---|",
    ]
    for cid in b.concepts:
        cb = b.concept_beliefs[cid]
        out.append(f"| {CONCEPTS[cid].name} | {diag.labels[cid]} | {cb.mean:.2f} ± {cb.sd:.2f} | {cb.asked} |")

    def evidence(mid):
        nums = [f"#{a.number}" for a in b.answers if a.attributed_misconception == mid]
        return ", ".join(nums)

    if diag.confirmed:
        out += ["", "### Misconceptions found"]
        for m in diag.confirmed:
            out.append(f"- **{MISCONCEPTIONS[m].description}** (evidence: {evidence(m)}; "
                       f"review {CONCEPTS[MISCONCEPTIONS[m].concept].source})")
    if diag.suspected:
        out += ["", "### Possible misconceptions (seen once, not confirmed)"]
        for m in diag.suspected:
            out.append(f"- {MISCONCEPTIONS[m].description} (evidence: {evidence(m)})")
    if diag.root_causes:
        out += ["", "### Likely root causes"]
        for c, p in diag.root_causes:
            out.append(f"- {CONCEPTS[c].name} looks weak, and so does its prerequisite "
                       f"**{CONCEPTS[p].name}**. Start with {CONCEPTS[p].source}.")
    weak = [c for c in diag.weak if not any(c == rc for rc, _ in diag.root_causes)]
    if weak:
        out += ["", "### Review first"]
        out += [f"- **{CONCEPTS[c].name}**: {CONCEPTS[c].source}" for c in weak]
    return "\n".join(out)


# ---------- simulated-student tab (streams step by step) ----------

def sim_question_html(step: TraceStep) -> str:
    q = step.question
    opts = "".join(f"<div>{LETTERS[i]}. {esc(o.text)}</div>" for i, o in enumerate(q.options))
    return (f"<div class='sb-card'><div class='sb-kicker'>Question {step.number} · {esc(CONCEPTS[q.concept].name)} · "
            f"L{q.level}</div><p>{esc(q.stem)}</p><div class='sb-note'>{opts}</div></div>")


def sim_answer_html(step: TraceStep, cause: str) -> str:
    q = step.question
    chosen = q.options[step.chosen_index]
    if step.correct:
        head = f"<span class='sb-pill ok'>correct</span> picked {LETTERS[step.chosen_index]} ({esc(step.confidence)})"
    else:
        tag = f" · option built on <code>{esc(chosen.misconception)}</code>" if chosen.misconception else " · untagged option"
        head = (f"<span class='sb-pill bad'>wrong</span> picked {LETTERS[step.chosen_index]} ({esc(step.confidence)}){tag}"
                f" · answer was {LETTERS[q.correct_index]}")
    parts = ["<div class='sb-card'><div class='sb-kicker'>Simulated student's answer</div>", f"<div>{head}</div>",
             f"<div class='sb-note'>hidden true cause (the agent can't see this): {esc(cause)}</div>"]
    if step.reading is not None:
        r = step.reading
        label = r.kind + (f": {r.misconception_id}" if r.misconception_id else "")
        parts.append(f"<p><b>Student's reason:</b> “{esc(r.text)}”<br><b>Model read it as:</b> "
                     f"<span class='sb-pill warn'>{esc(label)}</span> {esc(r.explanation)}</p>")
    parts.append("<div class='sb-note'>" + "<br>".join(esc(n) for n in step.notes) + "</div></div>")
    return "".join(parts)


def sim_trace_md(entries: List[str]) -> str:
    return "\n\n".join(entries) if entries else "_Steps will appear here as they happen._"


def sim_step_md(step: TraceStep, cause: Optional[str] = None) -> str:
    d = step.decision
    line = f"**#{step.number} {ACTION_LABELS.get(d.action, d.action)}**"
    if d.concept:
        line += f" → {d.concept} L{d.level}" + (f" (testing `{d.target}`)" if d.target else "")
    out = [line, f"- why: {d.rationale}"]
    out += [f"- guardrail: {n}" for n in d.guardrail_notes]
    if cause is not None:
        res = "correct" if step.correct else "wrong"
        m = step.question.options[step.chosen_index].misconception
        out.append(f"- student: {res}, {step.confidence}" + (f", picked `{m}`" if m else "") + f" (true cause: {cause})")
        if step.reading is not None:
            out.append(f'- reason: "{step.reading.text}" → read as {step.reading.kind}'
                       + (f" ({step.reading.misconception_id})" if step.reading.misconception_id else ""))
    return "\n".join(out)
