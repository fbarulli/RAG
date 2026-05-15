---
id: 3f5dbbd3b5
question: Model too big
sort_order: 9
---

If your model is too big for GitHub, one option is to compress the model using `joblib`. For example:

```python
joblib.dump(model, model_filename, compress=('zlib', 6))
```

This will use zlib to compress the model. Note that this process may take a few moments as the model is being compressed.