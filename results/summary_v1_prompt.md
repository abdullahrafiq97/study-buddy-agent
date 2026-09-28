| Policy | Misconceptions found (recall) | False misconception flags / session | Weak concepts found (recall) | False weak flags / session | Exactly right diagnosis | Avg questions |
|---|---|---|---|---|---|---|
| agent | 76% | 0.00 | 47% | 0.21 | 51% | 12.0 |
| heuristic | 70% | 0.00 | 67% | 0.12 | 55% | 12.0 |
| rule | 50% | 0.00 | 87% | 0.30 | 48% | 12.0 |
| random | 23% | 0.00 | 40% | 0.23 | 32% | 12.0 |

Exactly-right diagnosis rate by student:

| Student | agent | heuristic | rule | random |
|---|---|---|---|---|
| strong_all_round | 100% | 80% | 80% | 80% |
| swaps_sens_spec | 40% | 40% | 80% | 40% |
| prob_not_logodds | 60% | 40% | 40% | 60% |
| trusts_training_error | 75% | 100% | 40% | 0% |
| shrinkage_gap | 60% | 60% | 60% | 20% |
| weak_foundations | 60% | 80% | 80% | 20% |
| two_misconceptions | 20% | 20% | 0% | 0% |
| classification_struggler | 0% | 20% | 0% | 40% |

Agent guardrail rejections: 4, fallbacks to rule policy: 0, stopped early: 0/39 sessions
