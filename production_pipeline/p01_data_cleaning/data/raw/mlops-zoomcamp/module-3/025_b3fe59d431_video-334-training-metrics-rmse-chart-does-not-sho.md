---
id: b3fe59d431
question: 'Video 3.3.4: Training Metrics RMSE chart does not show due to the error:
  KeyError: ‘rmse_LinearRegression’'
sort_order: 25
---

Solution: Check the difference between xgboost and sklearn pipelines. In the xgboost pipeline, there is a `track_experiment` callback, which is missing in the sklearn pipeline.

Please add these lines:

You can refer to them in the similar commit linked here:

[Lines to be added](https://github.com/nilarte/mlops-zoomcamp-mage/commit/16c01dfcc2541a03a49f4744d0b0f0207c06e99d#diff-a97c890cfc31702e4b94f1e9a05558ebb8a57349fc9ccf153e82ae53af1bd53e)