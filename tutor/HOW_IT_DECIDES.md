### The loop

1. **Beliefs (deterministic math).** For each concept, mastery is a Beta distribution that starts at "no idea" (0.50 ± 0.29). For each misconception it counts *hits* (you picked the wrong option built on it) and *avoids* (you saw that option and didn't pick it).
2. **Decision (the agent).** The LLM reads the beliefs and your answer history, states what it currently thinks about you, and chooses one of seven moves: *screen*, *probe a misconception*, *step down*, *step up*, *recheck*, *check a prerequisite*, or *stop*. It also says what a right or a wrong answer would tell it.
3. **Guardrails.** Code checks the move before it runs. For example, it can't *step up* after a wrong answer, can't *stop* before the minimum number of questions, and can't re-probe a misconception that is already confirmed. A rejected move goes back to the LLM once with the reason; if the second try also fails, the session falls back to the fixed rule and says so.
4. **Question pick (deterministic).** An unasked question from the bank that matches the move.

### How evidence is weighed

| Answer | Mastery update |
|---|---|
| Correct at L1 / L2 / L3 | +0.5 / +1.0 / +1.5 toward mastery, scaled by confidence (sure ×1, unsure ×0.6, guess ×0.2) |
| Wrong at L1 / L2 / L3 | +1.5 / +1.0 / +0.5 toward "not mastered" |

**"Two wrong on topic A and one wrong on topic B: what does it conclude?"** That depends on *which* questions and *which* wrong answers:
- Two misses on easy (L1) questions count as 3.0 units of evidence. One miss on a hard (L3) question counts as 0.5. A is weak; B is still uncertain.
- If both A misses picked the *same misconception's* option with confidence, that misconception is **confirmed**. Those misses are then explained by the misconception rather than counted as a general gap in A.
- A miss marked "guess" on an untagged option points to a gap rather than a misconception.

**"What makes a question easier?"** Every question has a defined level. L1 is recall of a definition from the slides, L2 applies it to one given number or output, and L3 needs multi-step reasoning or judging a claim. *Stepping down* means asking a lower level on the same concept.

### Labels in the diagnosis

- **strong**: mastery ≥ 0.65 with at least 2 units of evidence
- **weak**: mastery ≤ 0.40 with at least 2 units of evidence, counting only wrong answers a confirmed misconception does not explain
- **misconception**: a misconception on this concept is confirmed (picked with a total hit weight ≥ 1.5, i.e. about twice)
- **uncertain**: not enough evidence either way

### Verification

The **Watch a simulated student** tab runs students with planted weaknesses. `python -m tutor.simulate` runs many sessions and compares the agent with a hand-written heuristic, the original ExamCraft rule, and random picks.
