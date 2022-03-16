# Benchmarking for Adversarial Error Debiasing

## Intro

Our proposed algorithm, Adversarial Error Debiasing (AED), will be evaluated against the below algorithms on the listed data sets. 

AED has two parameters that we should vary in the experiments:

- absolute: Whether the difference between the prediction and the true value should be taken as an absolute value or not.
- epsilon: We don't want the difference to be 0 because that zeros out the weights, instead we have a small offset, epsilon, to handle that.


## Datasets

### AIF360

- Adult/Census Dataset: 
- German Credit Data
- ProPublica COMPAS
- Bank Marketing
- Medical Expenditure Panel Survey Data

### NLP

- Our 4 dimensions
- Other (from Ahmed)

## Benchmark Algorithms

### Preprocessing

- DisparateImpactRemover
- LFR
- OptimPreproc
- Reweighting

### In-processing

- AdversarialDebiasing
- ARTClassifier
- GerryFairClassifier
- MetaFairClassifier
- PrejudiceRemover
- ExponentiatedGradientReduction
- GridSearchReduction

### Post-processing

- CalibratedEqOddsPostprocessing
- EqOddsPostprocessing
- RejectOptionClassification



