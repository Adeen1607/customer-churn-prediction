# Customer Churn Prediction

A reproducible telecom churn-classification project focused on validation discipline, class imbalance, threshold selection, interpretable evaluation, and retention-oriented decision support.

## Business objective

The model ranks customers by churn risk so a retention team can prioritize outreach. It does not automatically decide which customers should receive an offer. The decision threshold is selected against a minimum recall requirement and should ultimately be tied to campaign capacity, intervention cost, and expected retained value.

## Data source

The project uses the [Iranian Churn dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset). The dataset contains 3,150 telecom customer records collected over 12 months with behavioural and account attributes.

## Workflow

1. Fetch the official dataset through `ucimlrepo`.
2. Validate the target and remove duplicate rows.
3. Create stratified train, validation, and test partitions.
4. Fit preprocessing only on training data.
5. Compare logistic regression and random forest baselines.
6. Select the model using validation average precision.
7. Select a probability threshold under a minimum-recall constraint.
8. Evaluate the locked model and threshold once on the test set.
9. Export metrics, predictions, curves, and the fitted pipeline.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --minimum-recall 0.75
pytest
```

## Reported metrics

- churn prevalence;
- average precision and ROC AUC;
- precision, recall, and F1 at the selected threshold;
- false-positive and false-negative counts;
- confusion matrix and precision-recall curve.

## Responsible use

The sample comes from one telecom provider and one historical period. Churn interventions can affect customer experience and may not have equal benefit across groups. Deployment would require calibration, uplift testing, subgroup analysis, drift monitoring, contact-policy review, and human oversight.
