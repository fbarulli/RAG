---
id: e8dbc40d0c
question: X has 526 features, but expecting 525 features
sort_order: 19
---

Error:

```
ValueError: X has 526 features, but LinearRegression is expecting 525 features as input.
```

Solution:

The `DictVectorizer` creates an initial mapping for the features (columns). When calling the `DictVectorizer` again for the validation dataset, `transform` should be used as it will ignore features that it did not see when `fit_transform` was last called. For example:

```python
X_train = dv.fit_transform(train_dict)

X_test = dv.transform(test_dict)
```