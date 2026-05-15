---
id: 376ba649d7
question: 'AttributeError: module ''mlflow'' has no attribute ''set_tracking_url'''
sort_order: 23
---

In a mage block, the Python statement `mlflow.set_tracking_uri()` was returning an attribute error. This issue was observed when running Mage in one container and MLflow in another. If you encounter this, there may be something else in your project with the name "mlflow."

1. **Debugging the Import Issue:**
   - Insert a print statement before the Python statement that produces the attribute error:
     
     ```python
     print(mlflow.__file__)
     ```
     
   - This will show what the `mlflow` module points to. It should return a site-packages location, something like:
     
     ```
     '/usr/local/lib/python3.10/site-packages/mlflow/__init__.py'
     ```
     
   - If not, you may have another file or folder called "mlflow" that is confusing the Python import statement.

2. **Checking Backend Store Location:**
   - Look at the folder name where the `mlflow.db` is being created via this command (either in command line or in the Dockerfile for the MLflow service):
     
     ```bash
     mlflow server --backend-store-uri sqlite:///home/mlflow/mlflow.db --host 0.0.0.0 --port 5000
     ```
     
   - If the folder name for the backend store is "mlflow," Python may be trying to import that instead of the MLflow package you installed. Change the backend store folder name to something else, like `mlflow_data`.

   - Rename the folder in your local drive (since it gets mounted in `docker-compose.yml`).
   - Update the folder name in the Dockerfile for the MLflow service:
     - Specify the backend-store-uri in the MLflow server command with the new folder name.
   - Update the folder name in `docker-compose.yml` (when mounting the folder for the MLflow service), e.g.,
     
     ```yaml
     volumes:
       - "${PWD}/mlflow_data:/home/mlflow_data/"
     ```

3. **If `import mlflow` Gives a Module Not Found Error:**
   - Check the `PYTHONPATH` variable in the container:
     
     ```bash
     docker ps  # Copy the Mage container ID
     docker exec -it <container-ID> /bin/bash
     echo $PYTHONPATH
     ```
     
   - If you do not see the path to the site-packages directory for your Python version, add it to the `PYTHONPATH` environment variable.
   
   - To find out what path to use, execute this from the running container:
     
     ```python
     import sys
     print(sys.path)
     ```
     
   - Add this to the `PYTHONPATH` in the Dockerfile for the Mage service:
     
     ```dockerfile
     ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.10/site-packages"
     ```