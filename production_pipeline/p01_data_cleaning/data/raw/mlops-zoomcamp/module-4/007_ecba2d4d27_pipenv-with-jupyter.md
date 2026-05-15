---
id: ecba2d4d27
question: Pipenv with Jupyter
sort_order: 7
---

**Problem Description:** How can we use Jupyter notebooks with the Pipenv environment?

**Solution:**

1. Install Jupyter and `ipykernel` using Pipenv.

2. Register the kernel within the Pipenv shell using the following command:

   ```bash
   python -m ipykernel install --user --name=my-virtualenv-name
   ```

3. If you are using Jupyter notebooks in VS Code, this will also add the virtual environment to the list of kernels.

For more details, refer to this [Stack Overflow question](https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook).