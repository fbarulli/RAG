---
id: c6f2a2afe5
images:
- description: 'image #1'
  id: image_1
  path: images/machine-learning-zoomcamp/image_03f8bfd8.jpg
- description: 'image #2'
  id: image_2
  path: images/machine-learning-zoomcamp/image_a95d78a7.jpg
question: How to fix when %%capture output is not working in Google Collab Notebook
sort_order: 3
---

I was using Google Collab Notebook for the 2024 cohort HW 06. For Question 6, the following was not working in the Collab Notebook:

<{IMAGE:image_1}>

<{IMAGE:image_2}>

This led me to find a solution as follows:

1. **Import the required libraries:**

   ```python
   import io
   import sys
   ```

2. **Capture output using `io.StringIO`:**

   ```python
   output_capture = io.StringIO()
   sys.stdout = output_capture  # Redirect stdout to the StringIO buffer
   
   # Train the model with eta=0.3
   model_eta_03 = xgb.train(xgb_params, dtrain, num_boost_round=num_rounds, verbose_eval=2, evals=watchlist)
   
   # Reset stdout
   sys.stdout = sys.__stdout__
   
   # Retrieve and print the captured output
   captured_output = output_capture.getvalue()
   ```

3. **Modify the parser function for one line:**

   Replace this line in Alexey’s parser function:

   ```python
   for line in output.stdout.strip().split('\n'):
   ```

   With this line:

   ```python
   for line in output.strip().split('\n'):
   ```

4. **Call the parser function:**

   Use `df_score_03 = parse_xgb_output(captured_output)` to get the desired dataframe.