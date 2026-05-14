---
id: f7f0800d02
question: How to evaluate feature importance for numerical variables with AUC?
sort_order: 5
---

You can use the `roc_auc_score` function from the `sklearn.metrics` module. Pass the vector of the target variable (e.g., `above_average`) as the first argument and the vector of feature values as the second one. This function will return the AUC score for the feature that was passed as the second argument.

```python
from sklearn.metrics import roc_auc_score

# Example usage:
auc_score = roc_auc_score(target_variable, feature_values)
```
