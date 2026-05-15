---
id: 146d28247c
question: What is the difference between pandas get_dummies and sklearn OnehotEncoder?
sort_order: 18
---

They are basically the same. There are some key differences with regards to their input/output types, handling of missing values, etc., but they are both techniques to one-hot-encode categorical variables with identical results. 

- **pandas get_dummies**: A convenient choice when working with Pandas DataFrames.
- **sklearn OneHotEncoder**: More suitable for building a scikit-learn-based machine learning pipeline to handle categorical data as part of that pipeline.