---
id: af50bd4d07
question: 'Homework: Q3: Meaning of mean'
sort_order: 3
---

In question 3 of HW02 it is mentioned: ‘For computing the mean, use the training only’. What does that mean?

It means that you should use only the training data set for computing the mean, not the validation or test data set. This is how you can calculate the mean:

```python
# Calculate mean for a specific column in the training data
mean_value = df_train['column_name'].mean()
```

Another option:

```python
# Get descriptive statistics, including the mean
stats = df_train['column_name'].describe()
```
