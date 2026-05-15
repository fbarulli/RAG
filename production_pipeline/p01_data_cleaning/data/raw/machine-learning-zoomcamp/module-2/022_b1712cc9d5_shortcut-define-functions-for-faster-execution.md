---
id: b1712cc9d5
question: 'Shortcut: define functions for faster execution'
sort_order: 22
---

Defining functions can speed up development significantly. You can create a function like `prepare_df(initial_df, seed, fill_na_type)` to prepare all three dataframes and y_vectors. The `fillna()` operation can be applied before splitting the `initial_df`.

Additionally, you can reuse functions such as `rmse()` and `train_linear_regression(X, y, r)` from the class notebook.