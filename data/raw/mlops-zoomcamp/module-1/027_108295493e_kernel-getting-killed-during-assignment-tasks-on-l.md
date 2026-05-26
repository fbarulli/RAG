---
id: 108295493e
question: Kernel getting killed during assignment tasks on local
sort_order: 27
---

If the Jupyter notebook kernel gets killed repeatedly due to out-of-memory issues when converting a Pandas DataFrame to a dictionary or other memory-intensive steps, try using Google Colab as it offers more memory.

Here's how you can proceed:

1. **Upload the datasets** to Google Drive in the folder "Colab Notebooks."

2. **Mount the drive** on Colab:
   
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   
3. **Pull the data** from uploaded tables in Colab:

   ```python
   df_jan = pq.read_table('/content/drive/My Drive/Colab Notebooks/yellow_tripdata_2023-01.parquet').to_pandas()
   ```

4. **Complete the assignment** in Colab.

5. **Download the final assignment** to your local machine and copy it into the relevant repository.