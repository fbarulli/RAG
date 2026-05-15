---
id: 2eb2863faf
question: 'Why do I get a “ValueError: The truth value of a DataFrame is ambiguous.
  Use a.empty, a.bool(), a.item(), a.any() or a.all()” error when doing unit test
  that involves comparing two data frames?'
sort_order: 21
---

When you compare two Pandas DataFrames, the result is also a DataFrame. The same is true for Pandas Series. To properly compare them, you should not compare data frames directly.

Instead, convert the actual and expected DataFrames into a list of dictionaries, then use `assert` to compare the resulting lists.

Example:

```python
actual_df_list_dicts = actual_df.to_dict('records')

expected_df_list_dicts = expected_df.to_dict('records')

assert actual_df_list_dicts == expected_df_list_dicts
```