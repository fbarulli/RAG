---
id: 42e6f685af
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_c5935be1.png
- description: 'image #2'
  id: image_2
  path: images/mlops-zoomcamp/image_d1cb06e4.png
- description: 'image #3'
  id: image_3
  path: images/mlops-zoomcamp/image_54ddd284.png
question: Experiment not visible in MLflow UI
sort_order: 6
---



<{IMAGE:image_1}>

<{IMAGE:image_2}>

<{IMAGE:image_3}>

Make sure you launch the MLflow UI from the same directory as the code that is running the experiments (the same directory that contains the `mlruns` directory and the database that stores the experiments).

Or, navigate to the correct directory when specifying the `tracking_uri`.

For example:

- If the `mlflow.db` is in a subdirectory called `database`, the tracking URI would be:
  ```python
  sqlite:///database/mlflow.db
  ```
  
- If the `mlflow.db` is a directory above your current directory, the tracking URI would be:
  ```python
  sqlite:///../mlflow.db
  ```

Another alternative is to use an absolute path to `mlflow.db` rather than a relative path.

You can also launch the UI from the same notebook by executing the following code cell:

```python
import subprocess

MLFLOW_TRACKING_URI = "sqlite:///data/mlflow.db"

subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])
```

Then, use the same `MLFLOW_TRACKING_URI` when initializing MLflow or the client:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
```