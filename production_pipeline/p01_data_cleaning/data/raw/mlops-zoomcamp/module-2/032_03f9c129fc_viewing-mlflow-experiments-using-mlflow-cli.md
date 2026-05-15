---
id: 03f9c129fc
question: Viewing MLflow Experiments using MLflow CLI
sort_order: 32
---

Problem:

After starting the tracking server, when trying to use the MLflow CLI commands as listed [here](https://mlflow.org/docs/latest/cli.html), most commands can't find the experiments that have been run with the tracking server.

Solution:

- Set the environment variable `MLFLOW_TRACKING_URI` to the URI of the SQLite database:

  ```bash
  export MLFLOW_TRACKING_URI=sqlite:///{path to sqlite database}
  ```

- After setting the environment variable, you can view the experiments from the command line using commands like:

  ```bash
  mlflow experiments search
  ```

- Note: Commands like `mlflow gc` may still not get the tracking URI and need to be passed explicitly as an argument every time the command is run.