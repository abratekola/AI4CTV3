# AI4CT v0.3 Model Card

## Intended use
AI4CT supports clinical-trial operations teams with early identification of emerging site-level operational risk. It is designed for human-governed RBQM/central monitoring workflows.

## Not intended for
AI4CT is not a medical device, participant-level clinical prediction system, SAE causality assessor, DSMB replacement, regulatory decision-maker, or autonomous CAPA system.

## Predictive target
The supervised model estimates the probability that the next operational state will be high/critical, operationalized in local-history training as the next recorded overall operational-risk score being at least 50/100.

## Prediction horizons
The primary ML output is treated as a 30-day operational horizon in the interface. The 7- and 14-day values are derived by constant-hazard scaling of the 30-day probability. These shorter horizons are therefore not independently trained horizon-specific models in v0.3.

## Algorithm
- Random Forest classifier for supervised risk prediction
- Isolation Forest for unsupervised anomaly detection
- model-driver explanation from feature importance × distance from the learned feature baseline

## Features
Recruitment, screening failure, retention, missed visits, protocol deviations, CRF/query backlog, safety-reporting workflow, investigational-product stock/storage, laboratory turnaround/quality, staffing/training/turnover, monitoring actions, and selected changes over time.

## Training modes
### bootstrap-demo
When insufficient local longitudinal data exist, the model trains on deterministic synthetic operational scenarios. This allows the software workflow to function but is not evidence of predictive validity.

### local-history
When at least 40 site-period transitions are available, both outcome classes are represented, and each class has at least 8 observations, the model trains on this installation's longitudinal data.

These thresholds are software defaults, not evidence-based sample-size requirements for model development.

## Internal performance
When the training sample permits, AI4CT reports a stratified hold-out ROC AUC. This is an internal discrimination estimate only. It is not external validation and does not assess calibration, clinical/operational utility, fairness, or prospective impact.

## Explainability
AI4CT reports global Random Forest feature importance and site-specific driver scores based on feature importance multiplied by the current feature's distance from the learned baseline. These indicate model influence/association, not causality.

## Anomaly score
The Isolation Forest is fit to the same training distribution. The displayed anomaly score is the percentile of the current observation's anomaly statistic relative to the training distribution. It should be interpreted as unusualness, not probability of failure.

## Required validation before deployment
A serious evaluation should pre-specify:
- operational outcome definitions and prediction horizons
- temporal train/validation/test separation
- site/country external validation where possible
- discrimination (AUROC/AUPRC)
- calibration-in-the-large, calibration slope and calibration plots
- Brier score
- sensitivity/specificity and PPV/NPV at pre-specified alert thresholds
- alert burden and false-alert rate per site-period
- lead time gained
- net benefit / decision-curve or operational utility analysis
- subgroup/site/country performance
- missing-data sensitivity analyses
- model drift and recalibration rules
- human-factors/usability outcomes
- effect on monitoring/CAPA timeliness
- safety and governance review

## Versioning
Formal evaluations should freeze the feature definitions, preprocessing, training dataset, algorithm configuration, decision thresholds and software version. Any model update should create a new version and be revalidated.
