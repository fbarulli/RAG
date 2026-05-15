---
id: 1e05eabc5c
question: 'Mage: data_exporter block not taking all outputs from previous transformer
  block'
sort_order: 9
---

I encountered this issue while trying to run the `data_export` block that saves the dict vectorizer and the logs of the linear regression model into MLflow. My two distinct outputs were clearly created by the previous transformer block where the linear regression model is trained and the dict vectorizer is fitted to the training dataset.

I received this error while trying to run my export code:

```
Exception: Block mlflow_model_registry may be missing upstream dependencies. It expected to have 2 arguments, but only received 1. Confirm that the @data_exporter method declaration has the correct number of arguments.
```

The outputs are stored in a list, and this list is the input with the two outputs as the two elements. I had to modify my code in the `data_exporter` function to take only one argument and to define the two variables after that:

```python
Dv = output[0]

Lr = output[1]
```

This adjustment resolved the issue.