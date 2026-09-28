| Policy | Misconceptions found (recall) | False misconception flags / session | Weak concepts found (recall) | False weak flags / session | Exactly right diagnosis | Avg questions |
|---|---|---|---|---|---|---|
| agent+reasons | 57% | 0.00 | 57% | 0.09 | 51% | 12.0 |

Exactly-right diagnosis rate by student:

| Student | agent+reasons |
|---|---|
| strong_all_round | 90% |
| swaps_sens_spec | 40% |
| prob_not_logodds | 60% |
| trusts_training_error | 70% |
| shrinkage_gap | 50% |
| weak_foundations | 70% |
| two_misconceptions | 10% |
| classification_struggler | 20% |

Reason reader recovered the true cause in 94% of 264 informative reasons: guess 100% (n=137), slip 100% (n=50), misconception 79% (n=77)

Agent guardrail rejections: 15, fallbacks to rule policy: 1, stopped early: 0/80 sessions
