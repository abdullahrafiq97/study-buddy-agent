# tutor/concepts.py
"""
Concept map for the diagnostic session, built from Predictive Analytics
Lectures 3, 4, 5 (linear regression) and 8 (logistic regression).

Each concept lists its prerequisites and the specific misconceptions the
agent can test for. Every wrong option in the question bank is tagged with
one of these misconceptions (or with nothing, if it is a plain wrong answer),
so "which wrong answer did the student pick" becomes evidence.
"""
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Misconception:
    id: str
    concept: str
    description: str


@dataclass(frozen=True)
class Concept:
    id: str
    name: str
    prereqs: Tuple[str, ...]
    source: str
    misconceptions: Tuple[Misconception, ...]


def _m(concept: str, mid: str, description: str) -> Misconception:
    return Misconception(id=mid, concept=concept, description=description)


CONCEPT_LIST = [
    Concept(
        id="ols_basics",
        name="Simple regression & OLS",
        prereqs=(),
        source="Lecture 3 (slides 6-29), Lecture 4 (slide 31)",
        misconceptions=(
            _m("ols_basics", "nonlinear_x_means_nonlinear_model",
               "Thinks adding terms like X² makes a model non-linear, when 'linear' means linear in the parameters."),
            _m("ols_basics", "pvalue_as_effect_size",
               "Reads a small p-value as a large or important effect, rather than evidence that the effect is non-zero."),
            _m("ols_basics", "slope_means_strength",
               "Thinks a bigger slope means a stronger relationship, when the slope depends on units and strength is R²."),
        ),
    ),
    Concept(
        id="multiple_regression",
        name="Multiple regression interpretation",
        prereqs=("ols_basics",),
        source="Lecture 3 (slides 30-46), Lecture 4 (slides 18-28, 49-55)",
        misconceptions=(
            _m("multiple_regression", "ignores_holding_fixed",
               "Reads a multiple-regression coefficient as a total effect, forgetting it holds the other predictors fixed."),
            _m("multiple_regression", "collinearity_causes_bias",
               "Thinks collinearity biases coefficients (it only inflates standard errors); confuses it with omitted-variable bias."),
            _m("multiple_regression", "dummy_is_group_mean",
               "Reads a dummy coefficient as that group's mean, rather than its difference from the base group."),
        ),
    ),
    Concept(
        id="overfitting",
        name="Overfitting & bias-variance",
        prereqs=("ols_basics",),
        source="Lecture 4 (slide 9), Lecture 5 (slides 3-10, 33-38)",
        misconceptions=(
            _m("overfitting", "train_error_is_enough",
               "Believes the model with the lowest training error / highest R² will predict new data best."),
            _m("overfitting", "bias_variance_reversed",
               "Has the bias-variance direction backwards: thinks flexible models have high bias and low variance."),
            _m("overfitting", "validation_reuses_fit_data",
               "Thinks it is fine to evaluate or tune a model on data that was already used to fit or select it."),
        ),
    ),
    Concept(
        id="model_selection",
        name="Subset selection & shrinkage",
        prereqs=("overfitting", "multiple_regression"),
        source="Lecture 5 (slides 11-32, 39-49)",
        misconceptions=(
            _m("model_selection", "bigger_lambda_less_shrinkage",
               "Has the tuning parameter backwards: thinks a larger λ means less shrinkage / a model closer to OLS."),
            _m("model_selection", "ridge_zeroes_coefficients",
               "Thinks ridge regression sets coefficients exactly to zero (variable selection), confusing it with LASSO."),
            _m("model_selection", "no_standardization_needed",
               "Thinks ridge/LASSO are scale invariant like OLS, so predictors need not be standardized."),
        ),
    ),
    Concept(
        id="logistic",
        name="Logistic regression & odds",
        prereqs=("ols_basics",),
        source="Lecture 8 (slides 10-43)",
        misconceptions=(
            _m("logistic", "logit_coef_is_prob_change",
               "Reads a logit coefficient as a change in probability, rather than a change in log-odds."),
            _m("logistic", "odds_equals_probability",
               "Treats odds and probability as the same quantity (e.g., odds ratio 2 means probability doubles)."),
            _m("logistic", "constant_marginal_effect",
               "Thinks X changes the probability by the same amount for everyone, when in logit it is largest near P = 0.5."),
        ),
    ),
    Concept(
        id="classification",
        name="Classification thresholds & accuracy",
        prereqs=("logistic",),
        source="Lecture 8 (slides 46-63)",
        misconceptions=(
            _m("classification", "sensitivity_specificity_swap",
               "Swaps sensitivity (share of actual positives caught) and specificity (share of actual negatives caught)."),
            _m("classification", "threshold_direction",
               "Has the threshold effect backwards: thinks raising the threshold increases sensitivity."),
            _m("classification", "accuracy_on_rare_events",
               "Trusts total accuracy when the event is rare, even though predicting 'no event' for everyone scores high."),
        ),
    ),
]

CONCEPTS: Dict[str, Concept] = {c.id: c for c in CONCEPT_LIST}
MISCONCEPTIONS: Dict[str, Misconception] = {
    m.id: m for c in CONCEPT_LIST for m in c.misconceptions
}

LEVELS = {
    1: "L1 recall: state a definition or a fact from the slides",
    2: "L2 apply: interpret a given number or output in one step",
    3: "L3 analyze: multi-step reasoning, comparing scenarios, or judging a claim",
}


def dependents(concept_id: str) -> Tuple[str, ...]:
    """Concepts that list concept_id as a prerequisite."""
    return tuple(c.id for c in CONCEPT_LIST if concept_id in c.prereqs)
