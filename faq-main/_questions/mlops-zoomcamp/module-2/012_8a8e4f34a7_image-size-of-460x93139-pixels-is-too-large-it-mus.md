---
id: 8a8e4f34a7
question: Image size of 460x93139 pixels is too large. It must be less than 2^16 in
  each direction.
sort_order: 12
---

This issue is caused by `mlflow.xgboost.autolog()` in version 1.6.1 of XGBoost. To resolve this:

- Downgrade XGBoost to version 1.6.0 using the following command:

```bash
pip install xgboost==1.6.0
```

- Alternatively, update your requirements file to specify `xgboost==1.6.0`. 
