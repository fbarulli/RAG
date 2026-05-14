---
id: ac851e936b
question: Reading large parquet files
sort_order: 26
---

When reading large parquet files, you might encounter the following error:

```
IndexError: index 311297 is out of bounds for axis 0 with size 131743
```

Here are some possible solutions:

1. **Run as a Python Script:**
   - Try executing your code as a standalone Python script instead of within Jupyter Notebook.

2. **Use PySpark Library:**
   - Consider using the PySpark library, which is optimized for handling large data files.

3. **Read Parquet in Chunks:**
   - You can read parquet files in chunks using the pyarrow library. Reference this [blog post](http://blog.clairvoyantsoft.com/efficient-processing-of-parquet-files-in-chunks-using-pyarrow-b315cc0c62f9) for more details.

Using these methods may help manage and process large parquet files more efficiently.