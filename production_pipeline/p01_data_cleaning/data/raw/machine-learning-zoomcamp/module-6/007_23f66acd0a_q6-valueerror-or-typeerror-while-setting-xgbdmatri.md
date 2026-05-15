---
id: 23f66acd0a
question: 'Q6: ValueError or TypeError while setting xgb.DMatrix(feature_names=)'
sort_order: 7
---

If you’re encountering a **TypeError** like this:

```
TypeError: Expecting a sequence of strings for feature names, got: <class 'numpy.ndarray'>
```

This might be because you have executed:

```python
features = dv.get_feature_names_out()
```

This returns a `numpy.ndarray` instead of a list. Converting it to a list with `list(features)` won't solve the issue.


If you face a **ValueError** such as:

```
ValueError: feature_names must be string, and may not contain [, ] or <
```

This could be because you have tried one of these:

- `features = list(dv.get_feature_names_out())`
- `features = dv.feature_names_`

The problem originates from the output of `DictVectorizer`, which might look like:

```
['households',
 'housing_median_age',
 'latitude',
 'longitude',
 'median_income',
 'ocean_proximity=<1H OCEAN',
 'ocean_proximity=INLAND',
 'population',
 'total_bedrooms',
 'total_rooms']
```

The symbols `[, ]` or `<` are not compatible with XGBoost.

Solutions:

1. Do not specify `feature_names=` when creating `xgb.DMatrix`.
2. Alternatively, you can clean your feature names using regex:

    ```python
    import re
    features = dv.feature_names_
    pattern = r'[\[\]<>]'
    features = [re.sub(pattern, '  ', f) for f in features]
    ```