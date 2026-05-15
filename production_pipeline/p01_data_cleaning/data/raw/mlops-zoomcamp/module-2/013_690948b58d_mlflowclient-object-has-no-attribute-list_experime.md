---
id: 690948b58d
question: MlflowClient object has no attribute 'list_experiments'
sort_order: 13
---

Since version 1.29, the `list_experiments` method was deprecated and then removed in later versions.

You should use the following code instead:

```python
# Register the best model
model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.register_model(model_uri=model_uri, name="RandomForestBestModel")
```

For more details, refer to the [Mlflow documentation](https://mlflow.org/docs/1.29.0/python_api/mlflow.client.html#mlflow.client.MlflowClient.list_experiments).