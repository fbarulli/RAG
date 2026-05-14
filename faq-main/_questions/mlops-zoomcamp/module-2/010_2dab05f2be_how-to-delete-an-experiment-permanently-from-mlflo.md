---
id: 2dab05f2be
question: How to Delete an Experiment Permanently from MLFlow UI
sort_order: 10
---

After deleting an experiment from the UI, it may still persist in the database. To delete this experiment permanently, follow these steps:

1. **Install `ipython-sql`**
   
   ```bash
   pip install ipython-sql
   ```

2. **Load SQL Magic Scripts in Jupyter Notebook**

   ```python
   %load_ext sql
   ```

3. **Load the Database**
   
   Replace `nameofdatabase.db` with your actual database name:

   ```python
   %sql sqlite:///nameofdatabase.db
   ```

4. **Run SQL Script**

   Use SQL commands to delete the experiment permanently. Refer to [this link](https://stackoverflow.com/a/68431980/14151292) for a detailed guide.