# Study Buddy: a diagnostic quiz agent

A short quiz on regression and classification (Predictive Analytics, Lectures 3-5 and 8) where an **agent decides what to ask next**. It aims to work out, in about 12 questions, which concepts you have mastered, where you have a knowledge gap, and which *specific misconceptions* you hold. It shows its reasoning for every choice.

## Run it

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py                                   # the app, at http://localhost:7860
.venv/bin/python -m tutor.simulate --policies agent heuristic rule random --runs 5   # verification
.venv/bin/python -m tutor.simulate --policies agent --show two_misconceptions        # one full trace
```

`.env` needs `OPENAI_API_KEY`. Optional: `AGENT_MODEL` (default `gpt-5.4-mini`) and `AGENT_REASONING` (default `low`).

## What is agentic here (and what is not)

| Part | Deterministic or agent? |
|---|---|
| Question bank: 54 questions, 6 concepts × 3 misconceptions × 3 levels, with every wrong option tagged by the misconception it reveals | deterministic data (`tutor/data/question_bank.json`) |
| Belief update after each answer (Beta mastery, misconception hits/avoids) | deterministic math (`tutor/beliefs.py`) |
| **Choosing the next move** (screen / probe misconception / step down / step up / check prerequisite / stop) and explaining why | **LLM agent** (`tutor/agent.py`) |
| Guardrails that reject invalid moves, e.g. stepping up after a wrong answer, or stopping while a concept is still uncertain | deterministic (`tutor/policies.py`) |
| Final diagnosis | deterministic, from the beliefs |

## Files

- `tutor/concepts.py`: concept map, prerequisites, misconceptions (with lecture sources)
- `tutor/bank.py`: loads and validates the question bank
- `tutor/beliefs.py`: belief state and diagnosis
- `tutor/policies.py`: actions, guardrails, baselines (heuristic, ExamCraft v1 rule, random)
- `tutor/agent.py`: the LLM decision-maker
- `tutor/session.py`: the session loop and reasoning trace
- `tutor/simulate.py`: simulated students with planted weaknesses, plus the evaluation
- `app.py`: Gradio app
- `examcraft_v1/`: the original ExamCraft project (kept for reference)

## Verification results (12-question budget, 8 simulated students × 10 runs)

"Exactly right" = the diagnosis found every planted weakness and misconception, with no false flags.

| Who decides | Exactly right | Misconceptions found | Weak concepts found | False weak flags / session |
|---|---|---|---|---|
| LLM agent (prompt v2) | 62% | 68% | 57% | 0.06 |
| Hand-written heuristic (same actions) | 64% | 80% | 63% | 0.11 |
| ExamCraft v1 rule | 45% | 48% | 73% | 0.30 |
| Random | 28% | 20% | 27% | 0.17 |

- Agent prompt v1 (before the fixes): 51% exactly right, 0.21 false weak flags per session (5 runs; `results/summary_v1_prompt.md`).
- Agent vs heuristic, paired by student and seed: -1 point, 95% bootstrap CI [-14, +13]. **No measurable difference.**
- Both beat the original ExamCraft rule by about 18 points.

Full traces are in `results/runs_*.jsonl`.

## Free-text reasons (after a wrong answer, the student writes why)

The model reads the reason (misconception / gap / slip / guess / unclear). Tested on held-out seeds, with 30% of simulated students' confidence ratings random (real students are miscalibrated).

| Setup (same students and seeds, runs 100-109) | Exactly right | False misconception flags / session | False weak flags / session |
|---|---|---|---|
| Rules decide, model reads reasons | **60%** | 0.00 | 0.14 |
| Rules decide, no reasons | 58% | 0.03 | 0.15 |
| Model decides **and** reads reasons (agent: gpt-5-mini) | 51% | 0.00 | 0.09 |

- Over all 160 held-out sessions (runs 100-119), reasons improved the rules from 53% to 59% (+6.2 points, 95% CI [0, +13]).
- The reason reader (gpt-5.4-nano) recovered the true cause 94-96% of the time.
- Agent vs rules, both reading reasons: -8.7 points, 95% CI [-21, +4]. The model deciding does not help. Note: the agent here ran on gpt-5-mini (cheaper than the gpt-5.4-mini used earlier), so model strength is a possible confound.
- Perfect-reader ceiling (no LLM, `--oracle`): +2.5 points with honest confidence, +5.4 at 30% noise, +9.6 at 50%. **Reasons only help when stated confidence is unreliable.**

**Conclusion: the model earns its place reading what the student meant, not choosing the next question.**

Total API cost of all reason experiments: about $3.30.
