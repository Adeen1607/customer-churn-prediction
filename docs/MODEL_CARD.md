# Model card

## Intended use

Prioritize customers for human-reviewed retention outreach. Scores are risk rankings, not evidence that a particular offer will prevent churn.

## Validation design

- stratified 60% train, 20% validation, and 20% test;
- preprocessing learned only from training data;
- candidate selected by validation average precision;
- threshold selected on validation data under a minimum-recall constraint;
- locked model evaluated once on the held-out test partition.

## Models

The experiment compares class-weighted logistic regression with a class-weighted random forest. Logistic regression supplies a transparent linear baseline; random forest captures nonlinear interactions. Selection is based on validation evidence rather than training fit.

## Limitations

- one provider and one historical twelve-month period;
- no intervention cost or customer lifetime value;
- no randomized retention-treatment outcome;
- no guarantee that historical patterns remain stable;
- possible performance differences across customer groups;
- churn risk is not the same as persuadability.

## Production requirements

Production use would require probability calibration, time-based validation, uplift or treatment-effect testing, fairness and stability review, feature-availability checks, drift monitoring, campaign-capacity controls, and documented human oversight.
