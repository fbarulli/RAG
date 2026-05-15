---
id: 2ab69926dc
question: 'ValueError: Unknown label type: ''continuous'''
sort_order: 14
---

This problem occurs when using `DecisionTreeClassifier` instead of `DecisionTreeRegressor`.

To resolve this issue:

- Check whether you want to use a decision tree for classification or regression.
- Use `DecisionTreeRegressor` for regression tasks.

```python
from sklearn.tree import DecisionTreeRegressor

# Example: for regression
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
```