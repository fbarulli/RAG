---
id: ad15f21a9f
question: 'TypeError: send_file() unexpected keyword ''max_age'' during MLflow UI
  Launch'
sort_order: 27
---

**Problem:** When running `$ mlflow ui` on a remote server and attempting to open it in a local browser, the following exception occurs, and the MLflow UI page does not load.

**Solution:**

1. Uninstall Flask on your remote server by using:
   
   ```bash
   pip uninstall flask
   ```
   
2. Reinstall Flask with:
   
   ```bash
   pip install Flask
   ```
   
   This issue arises because the base conda environment includes a version of Flask that's less than 1.2. Cloning this environment retains the older version, causing the error. Installing a newer version of Flask resolves the issue.