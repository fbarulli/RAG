---
id: 7032c161df
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_168171a5.png
question: MLflow.xgboost Autolog Model Signature Failure
sort_order: 16
---

Got the same warning message as Warrie Warrie when using `mlflow.xgboost.autolog()`:

<{IMAGE:image_1}>

It turned out that this was just a warning message, and upon checking MLflow UI (making sure that no "tag" filters were included), the model was actually automatically tracked in MLflow.