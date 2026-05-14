---
id: 091cf6f73e
question: 'XGBoost: TypeError: Expecting a sequence of strings for feature names, got: <class ''numpy.ndarray''>'
sort_order: 1
---

This error occurs because recent versions of xgb.DMatrix expect the `feature_names` parameter to be a list of strings rather than a NumPy array. Older tutorial videos might use `feature_names=dv.get_feature_names_out()` directly, which now results in this error.

Convert `dv.get_feature_names_out()` to a list using `.tolist()`. Here's an updated example:

```python
# Convert feature names to a list
feature_names = dv.get_feature_names_out().tolist()

# Create DMatrix objects with the corrected feature names
dfulltrain = xgb.DMatrix(
    X_full_train, 
    label=y_full_train, 
    feature_names=feature_names
)

dtest = xgb.DMatrix(
    X_test, 
    feature_names=feature_names
)
```

**Explanation:** The `dv.get_feature_names_out()` method returns a NumPy array, but `xgb.DMatrix` now expects `feature_names` to be a list of strings. Using `.tolist()` converts the array into a compatible format, allowing the code to run without errors.