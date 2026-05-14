---
id: c6c716b9e7
question: How to copy a dataframe without changing the original dataframe?
sort_order: 19
---

Copy a dataframe using:

```python
X_copy = X.copy()
```

This creates a deep copy of the dataframe. If you use `X_copy = X`, it will create a "view" and any changes to `X_copy` will affect the original dataframe `X`. This is not a real copy.