---
id: 5a508f2d8c
question: 'Mage error: Error loading custom object at…'
sort_order: 31
---

When returning an object from a block, you may encounter an error like this:

```plaintext
Error loading custom_object at /home/src/mage_data/*************/pipelines/taxi_duration_pipe/.variables/make_predictions/output_0: [Errno 2] No such file or directory: '/home/src/mage_data/*************/pipelines/taxi_duration_pipe/.variables/make_predictions/output_0/object.joblib'

Error loading custom_object at /home/src/mage_data/*************/pipelines/taxi_duration_pipe/.variables/make_predictions/output_0: [Errno 2] No such file or directory: '/home/src/mage_data/*************/pipelines/taxi_duration_pipe/.variables/make_predictions/output_0/object.joblib'
```

This occurred when returning a `numpy.ndarray`, specifically the `y_pred` variable containing the predictions for the taxi dataset. It seems Mage struggles with some types of objects and expects data structures like DataFrames instead of `numpy.ndarrays`. To resolve this, you can return a DataFrame that includes both the `y_pred` and the ride IDs.