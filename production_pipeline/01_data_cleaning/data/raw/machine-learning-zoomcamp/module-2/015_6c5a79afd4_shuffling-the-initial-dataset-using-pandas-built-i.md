---
id: 6c5a79afd4
question: Shuffling the initial dataset using pandas built-in function
sort_order: 15
---

It is possible to shuffle the dataset using the pandas built-in function [`pandas.DataFrame.sample`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html). To shuffle the complete dataset and reset the index, use the following commands:

- Set `frac=1` to return a shuffled version of the complete dataset.
- Set `random_state=seed` for consistent randomization.

```python
df_shuffled = df.sample(frac=1, random_state=seed)
df_shuffled.reset_index(drop=True, inplace=True)
```