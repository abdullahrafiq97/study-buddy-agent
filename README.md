# Study Buddy: a diagnostic quiz agent

**The problem:** when I get a practice question wrong, I don't know *why*: a gap in what I know, a specific misconception, or a slip. Re-reading everything wastes the time I have before an exam.

**What the agent does:** it runs a short quiz (up to 12 questions) on regression and classification, the material from my Predictive Analytics course. After every answer it **decides what to ask next** to find out what I actually misunderstand, then reports it. After a wrong answer, it also asks me, before showing the right answer, to write one line on *why* I picked it, and uses that to tell a misconception from a guess or a slip.

## What the agent decides vs. what the code fetches

| Part | Who | File |
|---|---|---|
| **Choose the next move**: screen a new concept, probe a suspected misconception, step down or up a level, recheck, check a prerequisite, or stop | **LLM agent** | `tutor/agent.py` |
| **Read the student's one-line reason**: misconception / gap / slip / guess / unclear | **LLM** | `tutor/reasons.py` |
| Update scores after each answer (mastery per concept, evidence per misconception) | code (facts) | `tutor/beliefs.py` |
| Check the agent's move is legal (e.g. no stepping up after a wrong answer, no stopping while a concept is uncertain); a rejected move goes back to the agent once, then falls back to a rule | code (guardrails) | `tutor/policies.py` |
| Pick an unasked question that matches the move | code (fetch) | `tutor/policies.py`, `tutor/bank.py` |
| The loop: decide → ask → answer → update, capped at 12 questions | code | `tutor/session.py` |

No framework: the loop and the OpenAI API calls are my own code.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env

# Terminal: take the quiz yourself and watch the agent's trace live, step by step
.venv/bin/python -m tutor.cli                 # add --rules to let hand-written rules decide instead

# Terminal: watch the agent diagnose a simulated student, live (about $0.05)
.venv/bin/python -m tutor.simulate --policies agent+reasons --show swaps_sens_spec \
    --concepts logistic classification overfitting --max-q 12 --min-q 6 --confidence-noise 0.3

# App (http://localhost:7860): take the quiz yourself, or open "Watch a simulated student"
# to watch each decision stream live, with the planted weakness and the final verdict
.venv/bin/python app.py
```

Models (set in `.env` to override): agent `gpt-5.4-mini` (`AGENT_MODEL`), reason reader `gpt-5.4-nano` (`READER_MODEL`).

## How I tested it

Simulated students with **planted** weaknesses (e.g. "swaps sensitivity and specificity", "never learned ridge/LASSO", "shaky on OLS basics"). Each policy runs full sessions; "exactly right" means the final diagnosis found everything planted and flagged nothing false. `tutor/simulate.py` runs it; results and every trace are in `results/`.

**1. Does the model decide better than rules?** I wrote the same strategy as plain if-statements and compared (8 students × 10 runs, 12 questions):

| Who decides | Exactly right |
|---|---|
| LLM agent | 62% |
| Hand-written rules, same moves | 64% |
| My original ExamCraft rule (weakest topic, fixed thresholds) | 45% |
| Random | 28% |

Agent vs rules: -1 point, 95% CI [-14, +13], a tie. **Why:** every signal it read (which wrong option, stated confidence) was already a clean number, and a rule handles clean numbers fine.

**2. So I gave it a signal a rule can't read: the student's own words.** On fresh test runs, with 30% of the simulated students' confidence ratings unreliable:

| Same 80 sessions | Exactly right |
|---|---|
| Rules + the model reading reasons | **60%** |
| Rules, no reasons | 58% |
| Agent deciding + reading reasons (gpt-5-mini) | 51% |

- The reason reader recovered the true cause 94-96% of the time. Over 160 held-out sessions, reasons raised the rules from 53% to 59% (+6.2, 95% CI [0, +13]).
- With a *perfect* reader: +2.5 points when confidence is honest, +9.6 when half of it is random. **Reasons only matter when stated confidence is unreliable.**

## What doesn't work yet, and what I'd fix

- **The agent doesn't beat rules at choosing questions**, even with reasons (51% vs 60%, inside the noise, on a cheaper model). Its value is reading what the student meant.
- **The reason reader is a separate tool that concludes**, and writes its label into the scores before the agent sees it. The interpretation should be the agent's call: give it the raw words and let it decide.
- **The final diagnosis comes from fixed thresholds**, not the agent.
- **In traces, the agent sometimes trusts the confidence button over the student's words.** For example, the student clicked "Sure" but wrote "I just guessed", and the agent probed anyway (6 of 880 steps).
- **Simulated students are not real students.** Real results would need real users.

## Files

- `tutor/concepts.py`: 6 concepts, their prerequisites, and 3 misconceptions each
- `tutor/data/question_bank.json`: 54 questions (6 concepts × 3 misconceptions × 3 levels); every wrong option is tagged with the misconception it reveals
- `tutor/bank.py`: loads and validates the bank
- `tutor/beliefs.py`: score updates and the final diagnosis
- `tutor/policies.py`: the moves, guardrails, and the baseline policies
- `tutor/agent.py`: the LLM decision-maker
- `tutor/reasons.py`: the LLM that reads the student's reason
- `tutor/session.py`: the loop
- `tutor/cli.py`: take the quiz in the terminal; `tutor/live.py` prints each step as it happens
- `tutor/simulate.py`: simulated students and the evaluation
- `tutor/usage.py`: token and cost tracking
- `tutor/render.py`, `app.py`: the Gradio app (optional; the terminal command above is enough)

API cost of the free-text experiments: about $3.30.
