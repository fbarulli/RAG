---
id: 8231e2605d
question: 'MLflow container error: Can''t locate revision identified by …'
sort_order: 15
---

This means your MLflow container tries to access a db file which was a backend for a different MLflow version than the one you have in the container. Most likely, the MLflow version in the container does not match the MLflow version of the MLflow server you ran in module 2.

The easiest solution is to check which version you worked with before, and change the docker image accordingly.

1. Open a terminal on your host and activate the conda environment you worked in:

   ```bash
   conda activate <your-env-name>
   ```

2. Run the following command to check your MLflow version:

   ```bash
   mlflow --version
   ```

3. Edit the `mlflow.dockerfile` line to your version:

   ```Dockerfile
   RUN pip install mlflow==2.??.??
   ```

4. Save the file and rebuild the docker service by running:

   ```bash
   docker-compose build
   ```

5. Now you can start up the containers again, and your MLflow container should be able to successfully read your mounted DB file.