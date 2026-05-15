---
id: a0b5f19b14
question: Null column is appearing even if I applied .fillna()
sort_order: 9
---

When creating a duplicate of your dataframe, if you do the following:

```python
X_train = df_train
X_val = df_val
```

You're still referencing the original variable. This is called a shallow copy. To ensure that no references are attaching both variables and to keep a copy of the data, create a deep copy:

```python
X_train = df_train.copy()
X_val = df_val.copy()
```