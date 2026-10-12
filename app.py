# app.py
"""Study Buddy: a diagnostic quiz agent for regression & classification.

Run:  .venv/bin/python app.py
"""
import time

import gradio as gr

from tutor.agent import AgentError, AgentPolicy
from tutor.bank import load_bank
from tutor.concepts import CONCEPT_LIST
from tutor.policies import HeuristicPolicy, RandomPolicy, RuleBasedPolicy
from tutor.render import (
    CSS, beliefs_html, decision_html, diagnosis_md, feedback_md, option_choices, question_md, sim_answer_html,
    sim_question_html, sim_step_md, sim_trace_md, trace_md,
)
from tutor.session import Session
from tutor import simulate as sim
from tutor.simulate import PERSONAS, PERSONA_BY_NAME

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


def _reason_controls(show: bool, prompt: str = ""):
    # Toggled one by one: in Gradio 6.0.1 a hidden Column can't be hidden again after showing,
    # and a Textbox that starts hidden never appears. These all start visible and are hidden on load.
    return (gr.update(visible=show, value=prompt), gr.update(visible=show, value=""),
            gr.update(visible=show), gr.update(visible=show))


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
        *_reason_controls(False),
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
        *_reason_controls(False),
    )


def start(concepts, max_q, policy_label, ask_reasons):
    if not concepts:
        raise gr.Error("Pick at least one concept.")
    session = Session(POLICIES[policy_label](), concepts=concepts, max_questions=int(max_q),
                      min_questions=min(6, int(max_q)), bank=BANK, reasons=bool(ask_reasons))
    try:
        step = session.next_question()
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    return _show_question(session, step) if step else _show_end(session)


def _finalize(session, idx, conf, reason):
    try:
        step = session.submit(idx, conf, reason)
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    session.pending = None
    return (
        session,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=True),
        feedback_md(step),
        beliefs_html(session.beliefs),
        trace_md(session),
        *_reason_controls(False),
    )


def submit(session, answer, confidence):
    if session is None or session.current is None:
        raise gr.Error("Start a session first.")
    if not answer or not confidence:
        raise gr.Error("Pick an answer and how confident you are.")
    idx, conf = "ABCD".index(answer[0]), CONFIDENCE[confidence]
    if session.wants_reason(idx):
        # Ask before revealing the answer, so the reason isn't rewritten in hindsight.
        session.pending = (idx, conf)
        return (
            session,
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            gr.update(),
            gr.update(),
            *_reason_controls(True, f"**Before I show the answer: why did you pick {answer[0]}?** One line is enough."),
        )
    return _finalize(session, idx, conf, None)


def submit_reason(session, reason):
    if session is None or not getattr(session, "pending", None):
        raise gr.Error("Answer a question first.")
    if not reason or not reason.strip():
        raise gr.Error("Write a short reason, or press Skip.")
    return _finalize(session, *session.pending, reason)


def skip_reason(session):
    if session is None or not getattr(session, "pending", None):
        raise gr.Error("Answer a question first.")
    return _finalize(session, *session.pending, None)


def next_q(session):
    if session is None:
        raise gr.Error("Start a session first.")
    try:
        step = session.next_question()
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)
    return _show_question(session, step) if step else _show_end(session)


def simulate(persona_name, policy_label, read_reasons, noise, concepts, max_q, pause):
    """Stream a session with a simulated student: each event updates the page as it happens."""
    if not concepts:
        raise gr.Error("Pick at least one topic.")
    persona = PERSONA_BY_NAME[persona_name]
    sim.CONFIDENCE_NOISE = float(noise)
    policy = SIM_POLICIES[policy_label] + ("+reasons" if read_reasons else "")
    who = "The agent" if policy.startswith("agent") else "The rules"
    truth = persona.truth(concepts)
    header = (f"### Simulated student: {persona.name}\n{persona.description}  \n"
              f"**Planted (hidden from the agent):** weak = {sorted(truth['weak']) or 'none'} · "
              f"misconceptions = {sorted(truth['misconceptions']) or 'none'}")
    latest, previous, beliefs, entries, summary, session = "", "", "", [], "", None
    max_q = int(max_q)
    try:
        for event in sim.session_events(policy, persona, 0, max_q, min(6, max_q), BANK, concepts=concepts):
            kind = event[0]
            if kind == "start":
                session = event[1]
                beliefs = beliefs_html(session.beliefs)
            elif kind == "deciding":
                # Keep the previous step on screen while the next decision is made.
                banner = (f"<div class='sb-card'><div class='sb-kicker'>Step {event[1]} of up to {max_q}</div>"
                          f"<h4>⏳ {who} is deciding what to ask next… (previous step below)</h4></div>")
                latest = banner + previous
            elif kind == "decision":
                step = event[1]
                latest = decision_html(step) + (sim_question_html(step) if step.question else "")
                if step.question is None:  # the agent chose to stop
                    entries.append(sim_step_md(step))
                yield header, latest, beliefs, sim_trace_md(entries), summary
                time.sleep(float(pause))  # time to read the decision before the student answers
                continue
            elif kind == "answered":
                step, cause = event[1], event[2]
                latest = decision_html(step) + sim_question_html(step) + sim_answer_html(step, cause)
                previous = latest
                beliefs = beliefs_html(session.beliefs)
                entries.append(sim_step_md(step, cause))
            elif kind == "done":
                r = event[1]
                verdict = "✅ Exactly right" if r["exact"] else "❌ Not exactly right"
                summary = (f"## {verdict}\n"
                           f"**Planted:** weak = {sorted(truth['weak']) or 'none'} · misconceptions = "
                           f"{sorted(truth['misconceptions']) or 'none'}  \n"
                           f"**Found:** weak = {r['flagged_weak'] or 'none'} · misconceptions = "
                           f"{r['flagged_misconceptions'] or 'none'}"
                           + (f" · suspected = {r['suspected']}" if r["suspected"] else "")
                           + f"  \n{r['questions']} questions · {r['stop_reason']}")
            yield header, latest, beliefs, sim_trace_md(entries), summary
    except AgentError as e:
        raise gr.Error(_error(str(e)), duration=None)


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
                reasons_in = gr.Checkbox(value=True, label="Ask me why after a wrong answer",
                                         info="An LLM reads your one-line reason to tell a misconception from a gap, slip or guess.")
            start_btn = gr.Button("Start session", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                question_out = gr.Markdown("_Start a session to get your first question._")
                answer_in = gr.Radio(choices=[], label="Your answer", interactive=False)
                confidence_in = gr.Radio(list(CONFIDENCE), label="How confident are you?", interactive=False)
                with gr.Row():
                    submit_btn = gr.Button("Submit answer", variant="primary", interactive=False)
                    next_btn = gr.Button("Next question", interactive=False)
                reason_prompt = gr.Markdown()
                reason_in = gr.Textbox(show_label=False, lines=1, max_lines=2,
                                       placeholder="e.g. I thought a bigger λ means the model stays closer to OLS")
                with gr.Row():
                    reason_btn = gr.Button("Submit reason", variant="primary")
                    skip_btn = gr.Button("Skip")
                feedback_out = gr.Markdown()
            with gr.Column(scale=2):
                decision_out = gr.HTML(decision_html(None))
                beliefs_out = gr.HTML()
        diagnosis_out = gr.Markdown()
        with gr.Accordion("Full reasoning trace", open=False):
            trace_out = gr.Markdown()

        show_outputs = [session_state, question_out, answer_in, confidence_in, submit_btn, next_btn,
                        feedback_out, decision_out, beliefs_out, trace_out, diagnosis_out,
                        reason_prompt, reason_in, reason_btn, skip_btn]
        start_btn.click(start, [concepts_in, max_q_in, policy_in, reasons_in], show_outputs)
        next_btn.click(next_q, [session_state], show_outputs)
        answer_outputs = [session_state, answer_in, confidence_in, submit_btn, next_btn, feedback_out,
                          beliefs_out, trace_out, reason_prompt, reason_in, reason_btn, skip_btn]
        submit_btn.click(submit, [session_state, answer_in, confidence_in], answer_outputs)
        reason_btn.click(submit_reason, [session_state, reason_in], answer_outputs)
        reason_in.submit(submit_reason, [session_state, reason_in], answer_outputs)
        skip_btn.click(skip_reason, [session_state], answer_outputs)
        demo.load(lambda: _reason_controls(False), None, [reason_prompt, reason_in, reason_btn, skip_btn])

    with gr.Tab("Watch a simulated student"):
        gr.Markdown(
            "Each simulated student has a **known** weakness planted in it, hidden from the agent. "
            "Watch the agent decide, step by step, and check at the end whether it found what was planted. "
            "This is how the agent is verified."
        )
        with gr.Row():
            persona_in = gr.Dropdown([p.name for p in PERSONAS], value="swaps_sens_spec", label="Simulated student")
            sim_policy_in = gr.Radio(list(SIM_POLICIES)[:2], value="Agent (LLM decides)", label="Who decides")
            sim_max_in = gr.Slider(6, 18, value=12, step=1, label="Maximum questions")
        with gr.Row():
            sim_concepts_in = gr.CheckboxGroup(
                choices=[(c.name, c.id) for c in CONCEPT_LIST],
                value=["logistic", "classification", "overfitting"], label="Topics in this session",
            )
        with gr.Row():
            sim_reasons_in = gr.Checkbox(value=True, label="Student writes a reason after wrong answers")
            sim_noise_in = gr.Slider(0, 0.5, value=0.3, step=0.1, label="Unreliable confidence",
                                     info="Share of the student's confidence clicks that are random, like real students")
            sim_pause_in = gr.Slider(0, 8, value=3, step=1, label="Pause to read each decision (seconds)")
        with gr.Row():
            sim_btn = gr.Button("Run session", variant="primary")
            sim_stop = gr.Button("Stop")
        sim_header = gr.Markdown()
        sim_summary = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                sim_latest = gr.HTML()
            with gr.Column(scale=2):
                sim_beliefs = gr.HTML()
        with gr.Accordion("Full trace so far", open=False):
            sim_trace = gr.Markdown()
        run_event = sim_btn.click(
            simulate,
            [persona_in, sim_policy_in, sim_reasons_in, sim_noise_in, sim_concepts_in, sim_max_in, sim_pause_in],
            [sim_header, sim_latest, sim_beliefs, sim_trace, sim_summary],
        )
        sim_stop.click(None, None, None, cancels=[run_event])

    with gr.Tab("How it decides"):
        gr.Markdown(open("tutor/HOW_IT_DECIDES.md", encoding="utf-8").read())


if __name__ == "__main__":
    demo.launch(css=CSS)
