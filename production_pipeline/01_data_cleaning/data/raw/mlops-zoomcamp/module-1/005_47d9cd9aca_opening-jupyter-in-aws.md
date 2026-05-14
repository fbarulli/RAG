---
id: 47d9cd9aca
question: Opening Jupyter in AWS
sort_order: 5
---

Faced issue while setting up Jupyter Notebook on AWS. I was unable to access it from my desktop. (I am not using Visual Studio and hence faced problem)

1. Run the following command:
   
   ```bash
   jupyter notebook --generate-config
   ```

2. Edit the file `/home/ubuntu/.jupyter/jupyter_notebook_config.py` to add the following line:

   ```python
   NotebookApp.ip = '*'
   ```