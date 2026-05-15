---
id: 13fecb68a8
question: 'WARNING mlflow.xgboost: Failed to infer model signature: could not sample
  data to infer model signature: please ensure that autologging is enabled before
  constructing the dataset.'
sort_order: 39
---

Please make sure you follow the order below, enabling autologging before constructing the dataset. If you still have this issue, check that your data is in a format compatible with XGBoost.

1. **Enable MLflow autologging for XGBoost**
   
   ```python
   mlflow.xgboost.autolog()
   ```

2. **Construct your dataset**

   ```python
   X_train, y_train = ...
   ```

3. **Train your XGBoost model**

   ```python
   import xgboost as xgb
   model = xgb.XGBRegressor(...)
   
   model.fit(X_train, y_train)
   ```