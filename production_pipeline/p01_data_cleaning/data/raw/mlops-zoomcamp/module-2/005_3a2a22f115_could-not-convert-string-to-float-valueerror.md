---
id: 3a2a22f115
question: Could not convert string to float - ValueError
sort_order: 5
---

Running `python register_model.py` results in the following error:

```python
ValueError: could not convert string to float: '0 int\n1   float\n2     hyperopt_param\n3       Literal{n_estimators}\n4       quniform\n5         Literal{10}\n6         Literal{50}\n7         Literal{1}'
```

**Full Traceback:**

```python
Traceback (most recent call last):

File "/Users/name/Desktop/Programming/DataTalksClub/MLOps-Zoomcamp/2. Experiment tracking and model management/homework/scripts/register_model.py", line 101, in <module>

run(args.data_path, args.top_n)

File "/Users/name/Desktop/Programming/DataTalksClub/MLOps-Zoomcamp/2. Experiment tracking and model management/homework/scripts/register_model.py", line 67, in run

train_and_log_model(data_path=data_path, params=run.data.params)

File "/Users/name/Desktop/Programming/DataTalksClub/MLOps-Zoomcamp/2. Experiment tracking and model management/xfsub/scripts/register_model.py", line 41, in train_and_log_model

params = space_eval(SPACE, params)

File "/Users/name/miniconda3/envs/mlops-zoomcamp/lib/python3.9/site-packages/hyperopt/fmin.py", line 618, in space_eval

rval = pyll.rec_eval(space, memo=memo)

File "/Users/name/miniconda3/envs/mlops-zoomcamp/lib/python3.9/site-packages/hyperopt/pyll/base.py", line 902, in rec_eval

rval = scope._impls[node.name](*args, **kwargs)

ValueError: could not convert string to float: '0 int\n1   float\n2     hyperopt_param\n3       Literal{n_estimators}\n4       quniform\n5         Literal{10}\n6         Literal{50}\n7         Literal{1}'
```

**Solution:**

There are two plausible errors related to the `hpo.py` file where hyper-parameter tuning is run. The objective function should be structured as follows:

1. Ensure the `with` statement and the `log_params` function are correctly applied to log all runs and parameters:

    ```python
    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric('rmse', rmse)
    ```

2. Add the `with` statement immediately before the function, just after:

    ```python
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    ```

3. Log parameters just after defining the `search_space` dictionary:

    ```python
    search_space = {....}
    mlflow.log_params(search_space)
    ```

Logging parameters in groups can lead to issues because `register_model.py` expects to receive parameters individually. Ensure the objective function matches the example above.