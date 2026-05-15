---
id: 164b0e749d
question: Error raised when executing mlflow’s pyfunc.load_model in lambda function.
sort_order: 21
---

When the request is processed in a lambda function, the mlflow library raises the following warning:

```plaintext
2022/09/19 21:18:47 WARNING mlflow.pyfunc: Encountered an unexpected error (AttributeError("module 'dataclasses' has no attribute '__version__'")) while detecting model dependency mismatches. Set logging level to DEBUG to see the full traceback.
```

**Solution:**

- Increase the memory of the lambda function.