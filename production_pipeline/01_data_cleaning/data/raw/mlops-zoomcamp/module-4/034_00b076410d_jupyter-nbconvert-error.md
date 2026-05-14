---
id: 00b076410d
question: 'Jupyter: nbconvert error'
sort_order: 34
---

If you encounter an error when converting your `notebook.ipynb` into a Python script using the command:

```bash
jupyter nbconvert --to script your_notebook.ipynb
```

and you see the error message:

```bash
Jupyter command `jupyter-nbconvert` not found.
```

follow these steps:

1. **Verify the Directory**
   
   Ensure that you're in the directory containing your Jupyter notebook.
   
2. **Install the Necessary Package**

   If the issue persists, you may need to install the `nbconvert` package. Run the following command:
   
   ```bash
   pip install nbconvert
   ```

3. **Convert the Notebook**

   After installing `nbconvert`, use the following command to convert your notebook to a Python script:
   
   ```bash
   jupyter nbconvert your_notebook.ipynb --to python
   ```
   
   **Note:** The correct command is slightly different (`--to python` instead of `--to script`).