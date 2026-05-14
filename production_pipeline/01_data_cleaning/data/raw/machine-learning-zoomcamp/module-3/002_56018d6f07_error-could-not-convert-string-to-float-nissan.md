---
id: 56018d6f07
question: 'Error: Could not convert string to float: ''Nissan'''
sort_order: 2
---

The error message "could not convert string to float: 'Nissan'" typically occurs when a machine learning model or function is expecting numerical input but receives a string instead. In this case, it seems like the model is trying to convert the car brand 'Nissan' into a numerical value, which isn’t possible.

To resolve this issue, you can encode categorical variables like car brands into numerical values. One common method is one-hot encoding, which creates new binary columns for each category/label present in the original column.

Here’s an example of how you can perform one-hot encoding using pandas:

```python
import pandas as pd

# Assuming 'data' is your DataFrame and 'brand' is the column with car brands
data_encoded = pd.get_dummies(data, columns=['brand'])
```

In this code, `pd.get_dummies()` creates a new DataFrame where the `brand` column is replaced with binary columns for each brand (e.g., `brand_Nissan`, `brand_Toyota`, etc.). Each row in the DataFrame has a 1 in the column that corresponds to its brand and 0 in all other brand columns.