---
id: 601b847b43
question: Unsupported Scikit-Learn version
sort_order: 30
---

Getting the following warning when running `mlflow.sklearn`:

```
2022/05/28 04:36:36 WARNING mlflow.utils.autologging_utils: You are using an unsupported version of sklearn. If you encounter errors during autologging, try upgrading / downgrading sklearn to a supported version, or try upgrading MLflow.
```

**Solution:**

- Use scikit-learn version between 0.24.1 and 1.4.2.

**Reference:** [MLflow Documentation](https://www.mlflow.org/docs/latest/python_api/mlflow.sklearn.html)