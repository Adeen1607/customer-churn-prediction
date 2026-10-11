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


## Rebuild the reading edition

```bash
python -m pip install -r requirements-research.txt
python scripts/build_research_site.py
```

This renders saved notebook outputs without rerunning model training or external APIs. To refresh findings, run the analysis commands above first, update `reports/research-findings.md`, and rebuild.


<!-- research-edition:start -->
## Research edition

[Research report and final visual results](research/report.md) · [Browser-ready notebook catalogue](research/notebooks.html)

Fresh execution on 11 October 2026 using the committed pipeline and seed 42. Validation selected **random_forest**. On **570** held-out records, precision was **85.9%**, recall **88.8%** and average precision **0.9198**. The test set contained **10** missed positives and **13** false alerts. These are retrospective sample results.

| Measure | Value |
|---|---:|
| average precision | 0.9198 |
| roc auc | 0.9848 |
| precision | 0.8587 |
| recall | 0.8876 |
| f1 | 0.8729 |
| true negatives | 468 |
| false positives | 13 |
| false negatives | 10 |
| true positives | 79 |
| selected model | random_forest |
| decision threshold | 0.5795 |
| validation precision at threshold | 0.8375 |
| validation recall at threshold | 0.7528 |
| validation average precision | 0.8589 |
| churn prevalence | 0.1565 |
| train rows | 1710 |
| validation rows | 570 |
| test rows | 570 |
| random seed | 42 |



The `research/` folder contains a professional static reading edition. The Pages workflow deploys it after GitHub Pages is enabled with **GitHub Actions** as the publishing source. Intended URL: `https://adeen1607.github.io/customer-churn-prediction/`. Deployment status must be checked in Actions before treating that address as live.

<!-- research-edition:end -->
