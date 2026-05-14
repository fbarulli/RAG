---
id: d94e5cc566
question: 'Q6: Logged model artifacts lost when mlflow container is down or removed'
sort_order: 19
---

By default, the logged model and artifacts are stored in a local folder in the mlflow container but not in `/home/src/mlflow`. Therefore, when the container is restarted (after a compose down or container removal), the artifacts are deleted and you cannot see them in the mlflow UI.

To prevent this issue, you can include a new volume in the Docker Compose service for mlflow to map a folder on the local machine to the folder `/mlartifacts` in the mlflow container:

- ```bash
  "${PWD}/mlartifacts:/mlartifacts/"
  ```

This way, every data logged to the experiment will be available even when the mlflow container is recreated.