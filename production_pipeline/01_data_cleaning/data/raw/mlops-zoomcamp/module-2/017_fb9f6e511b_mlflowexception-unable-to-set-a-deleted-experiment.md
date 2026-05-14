---
id: fb9f6e511b
question: 'MlflowException: Unable to Set a Deleted Experiment'
sort_order: 17
---

```python
raise MlflowException(

mlflow.exceptions.MlflowException: Cannot set a deleted experiment 'random-forest-hyperopt' as the active experiment. You can restore the experiment, or permanently delete the experiment to create a new one.
```

To resolve this issue, consider the following options:

- **Restore or Permanently Delete the Experiment**: Refer to guidance on [Stack Overflow](https://stackoverflow.com/questions/60088889/how-do-you-permanently-delete-an-experiment-in-mlflow) for methods to permanently delete an experiment in MLflow.

- **Command Line Resolution**: If you have deleted the experiment from the MLflow UI, run the following command in the CLI. Make sure to use the correct database filename.
  
  ```bash
  mlflow gc --backend-store-uri sqlite:///backend.db
  ```

- **Ensure .trash is Empty**: If the above command does not work and your .trash folder is already empty, confirm this by executing:

  ```bash
  rm -rf mlruns/.trash/*
  ```
  
  Note: Ensure no files remain in `.trash/` that could be interfering with the experiment reset.