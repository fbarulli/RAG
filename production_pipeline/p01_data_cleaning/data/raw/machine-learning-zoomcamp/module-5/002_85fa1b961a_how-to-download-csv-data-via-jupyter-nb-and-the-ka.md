---
id: 85fa1b961a
question: How to download CSV data via Jupyter NB and the Kaggle API, for one seamless
  experience
sort_order: 2
---

To download CSV data via Jupyter Notebook using the Kaggle API, follow these steps:

1. **Set up a Kaggle account**:
   - Go to your Kaggle account settings, navigate to the API section, and click `Create New Token`. This will download a `kaggle.json` file containing your `username` and `key`.

2. **Place the `kaggle.json` file**:
   - Ensure the `kaggle.json` file is in the same directory as your Jupyter Notebook.

3. **Set permissions for the `kaggle.json` file**:
   ```bash
   !chmod 600 <ENTER YOUR FILEPATH>/kaggle.json
   ```

4. **Configure the environment**:
   - Import the `os` module and set the Kaggle config directory:
   ```python
   import os
   os.environ['KAGGLE_CONFIG_DIR'] = '<STRING OF YOUR FILE PATH>'
   ```

5. **Download the dataset**:
   - Use the Kaggle API to download your desired dataset:
   ```bash
   !kaggle datasets download -d kapturovalexander/bank-credit-scoring
   ```

6. **Unzip and access the CSV file**:
   ```bash
   !unzip -o bank-credit-scoring.zip
   ```

Follow these steps to seamlessly integrate Kaggle data retrieval into your Jupyter workflow.