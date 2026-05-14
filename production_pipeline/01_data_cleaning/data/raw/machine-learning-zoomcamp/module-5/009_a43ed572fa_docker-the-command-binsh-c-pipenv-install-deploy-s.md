---
id: a43ed572fa
question: 'Docker: The command ''/bin/sh -c pipenv install --deploy --system && rm
  -rf /root/.cache'' returned a non-zero code: 1'
sort_order: 9
---

After using the command `docker build -t churn-prediction .` to build the Docker image, this error occurs, and the image is not created.

To fix this issue, adjust the Python version in your Dockerfile to match the version installed on your system:

1. Determine your Python version by running:
   
   ```bash
   python --version
   ```
   
   Example output:
   
   ```bash
   Python 3.9.7
   ```

2. Update the first line of your Dockerfile with the correct Python version:

   ```dockerfile
   FROM python:3.9.7-slim
   ```

Make sure to replace `3.9.7` with your actual Python version.