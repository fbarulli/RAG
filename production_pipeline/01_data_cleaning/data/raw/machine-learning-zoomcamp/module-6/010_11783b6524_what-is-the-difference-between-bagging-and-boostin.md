---
id: 11783b6524
question: What is the difference between bagging and boosting?
sort_order: 10
---

For ensemble algorithms during week 6, one bagging algorithm and one boosting algorithm were presented: Random Forest and XGBoost, respectively.

Random Forest trains several models in parallel. The output can be, for example, the average value of all the outputs of each model. This is called bagging.

XGBoost trains several models sequentially: the previous model error is used to train the following model. Weights are used to ponderate the models so that the best models have higher weights and are therefore favored for the final output. This method is called boosting.

Note that boosting is not necessarily better than bagging.

Bagging stands for “Bootstrap Aggregation”:

- It involves taking multiple samples with replacement to derive multiple training datasets from the original training dataset (bootstrapping).
- A classifier (e.g., decision trees or stumps for Random Forests) is trained on each such training dataset.
- The predictions are combined (aggregation) to obtain the final prediction.
  - For classification, predictions are combined via voting; for regression, via averaging.
- Bagging can be done in parallel since the various classifiers are independent.
- Bagging decreases variance (but not bias) and is robust against overfitting.

Boosting, on the other hand, is sequential:

- Each model learns from the mistakes of its predecessor.
- Observations are given different weights - observations/samples misclassified by the previous classifier are given a higher weight.
- This process is continued until a stopping condition is reached (e.g., a maximum number of models is reached, or error is acceptably small).
- Boosting reduces bias and is generally more accurate than bagging, but can be prone to overfitting.

