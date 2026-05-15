---
id: c2f713127c
question: How to get feature importance for XGboost model
sort_order: 28
---

Using `model.feature_importances_` can give you an error:

```
AttributeError: 'Booster' object has no attribute 'feature_importances_'
```

If you train the model like this: `model = xgb.train`, you should use `get_score()` instead.