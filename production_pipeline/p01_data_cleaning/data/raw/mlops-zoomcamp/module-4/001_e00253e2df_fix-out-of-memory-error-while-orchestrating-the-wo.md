---
id: e00253e2df
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_d1d6da8a.png
- description: 'image #2'
  id: image_2
  path: images/mlops-zoomcamp/image_23d027f2.png
- description: 'image #3'
  id: image_3
  path: images/mlops-zoomcamp/image_09a24598.png
question: Fix Out of Memory error while orchestrating the workflow on a ML Pipeline
  for a high volume dataset.
sort_order: 1
---

We come across situations in data transformation & pre-processing as well as model training in a ML pipeline where we need to handle datasets of high dimensionality or high cardinality (usually millions). We often end up with Out of Memory (OOM) errors like below when the flow is running:

<{IMAGE:image_1}>

If you do not have the option of increasing your RAM, the following approaches can be effective in mitigating this error:

1. **Read Only Required Features/Columns:**
   - During the data loading step, read only the necessary features/columns from the dataset.
   
   <{IMAGE:image_2}>

2. **Remove Unused Dataframes:**
   - Before encoding/vectorizing, remove the dataframe when you have obtained `X_train` & `y_train`.
   
   <{IMAGE:image_3}>

3. **Create or Resize Swap File:**
   - If you do not have a swap file or have a small one, create a swap file (size as per memory requirement) or replace the existing one with a properly sized one.
   
   To remove an existing swapfile, use:
   
   ```bash
   sudo swapoff /swapfile
   sudo rm /swapfile
   ```
   
   To create a new properly sized swapfile (e.g., 16 GB), use:
   
   ```bash
   sudo fallocate -l 16G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```
   
   To check the swap file created:
   
   ```bash
   free -h
   ```