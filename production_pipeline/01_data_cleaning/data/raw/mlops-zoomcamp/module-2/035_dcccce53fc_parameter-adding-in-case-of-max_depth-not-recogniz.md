---
id: dcccce53fc
question: Parameter adding in case of max_depth not recognized
sort_order: 35
---

**Problem:** Parameter was not recognized during the model registry.

**Solution:** Parameters should be added prior to the model registry. Use the following method to add parameters:

```python
mlflow.log_params(params)
```

This way, the dictionary can be directly appended to `data.run.params`.  