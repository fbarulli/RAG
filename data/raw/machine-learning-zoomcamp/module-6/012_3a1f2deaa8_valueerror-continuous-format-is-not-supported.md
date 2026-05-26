---
id: 3a1f2deaa8
question: 'ValueError: continuous format is not supported'
sort_order: 12
---

Calling `roc_auc_score()` to get AUC is throwing the above error.

Solution to this issue is to ensure that you pass `y_actuals` as the first argument and `y_pred` as the second argument.

```python
roc_auc_score(y_train, y_pred)
```