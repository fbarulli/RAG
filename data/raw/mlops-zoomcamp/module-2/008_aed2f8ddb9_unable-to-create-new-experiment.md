---
id: aed2f8ddb9
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_d1cb06e4.png
question: Unable to create new Experiment
sort_order: 8
---

Following the instructions in the video did not work, even though the Jupyter notebook indicates it was successfully created.

<{IMAGE:image_1}>

It is recommended to set the URI to the listener directly. This discrepancy might be due to differences in the "mlflow" package versions between the video and the latest version we are using. The documentation for the latest "mlflow" package suggests setting the URI as follows:

```python
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
```