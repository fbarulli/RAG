---
id: d781b23627
question: Standard deviation using Pandas built in Function
sort_order: 26
---

In pandas, you can use the built-in function `std()` to calculate the standard deviation. For example:

- To get the standard deviation of a single column:

  ```python
  df['column_name'].std()
  ```

- To get the standard deviation of multiple columns:

  ```python
  df[['column_1', 'column_2']].std()
  ```
