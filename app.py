# app.py
"""Study Buddy: a diagnostic quiz agent for regression & classification.

Run:  .venv/bin/python app.py
"""
import gradio as gr

from tutor.agent import AgentError, AgentPolicy
from tutor.bank import load_bank
from tutor.concepts import CONCEPT_LIST
from tutor.policies import HeuristicPolicy, RandomPolicy, RuleBasedPolicy
from tutor.render import (
    CSS, beliefs_html, decision_html, diagnosis_md, feedback_md, option_choices, question_md, trace_md,
)
from tutor.session import Session
from tutor.simulate import PERSONAS, PERSONA_BY_NAME, run_one

BANK = load_bank()

POLICIES = {
    "Agent (LLM decides)": AgentPolicy,
    "Heuristic (fixed if-statements)": HeuristicPolicy,
    "ExamCraft v1 rule (weakest topic)": RuleBasedPolicy,
    "Random": RandomPolicy,
}
SIM_POLICIES = {"Agent (LLM decides)": "agent", "Heuristic (fixed if-statements)": "heuristic",
                "ExamCraft v1 rule (weakest topic)": "rule", "Random": "random"}
CONFIDENCE = {"Sure": "sure", "Unsure": "unsure", "Guessing": "guess"}


def _error(msg: str):
    return f"⚠️ {msg}\n\nIf the LLM is unavailable (e.g. no API credits), pick the **Heuristic** decision-maker to keep going."


def _show_question(session, step):
    return (
        session,
        question_md(step, session.ctx.max_questions),
        gr.update(choices=option_choices(step), value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(interactive=True),   # submit
        gr.update(interactive=False),  # next
        "",
        decision_html(step),
        beliefs_html(session.beliefs),
        trace_md(session),
        "",
    )


def _show_end(session, feedback=""):
    last = session.trace[-1] if session.trace else None
    return (
        session,
        "#### Session complete",
        gr.update(choices=[], value=None, interactive=False),
        gr.update(value=None, interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        feedback,
        decision_html(last),
        beliefs_html(session.beliefs),
        trace_md(session),
        diagnosis_md(session),
    )


def start(concepts, max_q, policy_label):
    if not concepts:
        raise gr.Error("Pick at least one concept.")
    session = Session(POLICIES[policy_label](), concepts=concepts, max_questions=int(max_q),
                      min_questions=min(6, int(max_q)), bank=BANK)
    try:
        step = session.next_question()
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    return _show_question(session, step) if step else _show_end(session)


def submit(session, answer, confidence):
    if session is None or session.current is None:
        raise gr.Error("Start a session first.")
    if not answer or not confidence:
        raise gr.Error("Pick an answer and how confident you are.")
    step = session.submit("ABCD".index(answer[0]), CONFIDENCE[confidence])
    return (
        session,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        feedback_md(step),
        beliefs_html(session.beliefs),
        trace_md(session),
    )


def next_q(session):
    if session is None:
        raise gr.Error("Start a session first.")
    try:
        step = session.next_question()
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    return _show_question(session, step) if step else _show_end(session)


def simulate(persona_name, policy_label, max_q):
    persona = PERSONA_BY_NAME[persona_name]
    try:
        r = run_one(SIM_POLICIES[policy_label], persona, 0, int(max_q), min(6, int(max_q)), BANK)
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    truth = persona.truth([c.id for c in CONCEPT_LIST])
    lines = [
        f"### {persona.name}: {persona.description}",
        f"**Planted:** weak = {sorted(truth['weak']) or 'none'} · misconceptions = {sorted(truth['misconceptions']) or 'none'}",
        f"**Found:** weak = {r['flagged_weak'] or 'none'} · misconceptions = {r['flagged_misconceptions'] or 'none'}"
        + (f" · suspected = {r['suspected']}" if r["suspected"] else ""),
        f"**Exactly right:** {'yes' if r['exact'] else 'no'} · {r['questions']} questions · {r['stop_reason']}",
        "",
        "---",
    ]
    for t in r["trace"]:
        head = f"**#{t['n']} {t['action']}**"
        if t["concept"]:
            head += f" → {t['concept']} L{t['level']}" + (f" (testing `{t['target']}`)" if t["target"] else "")
        lines.append(head)
        if t["hypothesis"]:
            lines.append(f"- read: {t['hypothesis']}")
        lines.append(f"- why: {t['rationale']}")
        lines += [f"- guardrail: {n}" for n in t["guardrail_notes"]]
        if t["question"]:
            res = "correct" if t["correct"] else "wrong" + (f", picked `{t['picked_misconception']}`" if t["picked_misconception"] else "")
            lines.append(f"- student: {res} ({t['confidence']})")
        lines.append("")
    return "\n".join(lines)


with gr.Blocks(title="Study Buddy: diagnostic quiz agent") as demo:
    gr.Markdown(
        "# Study Buddy\n"
        "A diagnostic quiz on **regression & classification** (Predictive Analytics, Lectures 3-5 and 8). "
        "After each answer, the agent decides what to ask next to work out what you know, where you have gaps, "
        "and which specific misconceptions you hold. It shows its reasoning for every choice."
    )
    session_state = gr.State(None)

    with gr.Tab("Take the quiz"):
        with gr.Accordion("Session setup", open=True):
            with gr.Row():
                concepts_in = gr.CheckboxGroup(
                    choices=[(c.name, c.id) for c in CONCEPT_LIST],
                    value=[c.id for c in CONCEPT_LIST], label="Concepts",
                )
            with gr.Row():
                max_q_in = gr.Slider(6, 18, value=12, step=1, label="Maximum questions")
                policy_in = gr.Radio(list(POLICIES), value="Agent (LLM decides)", label="Who decides the next question")
            start_btn = gr.Button("Start session", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                question_out = gr.Markdown("_Start a session to get your first question._")
                answer_in = gr.Radio(choices=[], label="Your answer", interactive=False)
                confidence_in = gr.Radio(list(CONFIDENCE), label="How confident are you?", interactive=False)
                with gr.Row():
                    submit_btn = gr.Button("Submit answer", variant="primary", interactive=False)
                    next_btn = gr.Button("Next question", interactive=False)
                feedback_out = gr.Markdown()
            with gr.Column(scale=2):
                decision_out = gr.HTML(decision_html(None))
                beliefs_out = gr.HTML()
        diagnosis_out = gr.Markdown()
        with gr.Accordion("Full reasoning trace", open=False):
            trace_out = gr.Markdown()

        show_outputs = [session_state, question_out, answer_in, confidence_in, submit_btn, next_btn,
                        feedback_out, decision_out, beliefs_out, trace_out, diagnosis_out]
        start_btn.click(start, [concepts_in, max_q_in, policy_in], show_outputs)
        next_btn.click(next_q, [session_state], show_outputs)
        submit_btn.click(
            submit, [session_state, answer_in, confidence_in],
            [session_state, answer_in, confidence_in, submit_btn, next_btn, feedback_out, beliefs_out, trace_out],
        )

    with gr.Tab("Watch a simulated student"):
        gr.Markdown(
            "Each simulated student has a **known** weakness planted in it. Run a session and check whether "
            "the decision-maker finds it. This is how the agent is verified."
        )
        with gr.Row():
            persona_in = gr.Dropdown([p.name for p in PERSONAS], value="two_misconceptions", label="Simulated student")
            sim_policy_in = gr.Radio(list(SIM_POLICIES), value="Agent (LLM decides)", label="Decision-maker")
            sim_max_in = gr.Slider(6, 18, value=12, step=1, label="Maximum questions")
        sim_btn = gr.Button("Run session", variant="primary")
        sim_out = gr.Markdown()
        sim_btn.click(simulate, [persona_in, sim_policy_in, sim_max_in], sim_out)

    with gr.Tab("How it decides"):
        gr.Markdown(open("tutor/HOW_IT_DECIDES.md", encoding="utf-8").read())


if __name__ == "__main__":
    demo.launch(css=CSS)
