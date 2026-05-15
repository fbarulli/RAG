---
id: 0d7b521e27
question: Sparse matrix compared to dense matrix
sort_order: 14
---

A sparse matrix is more memory-efficient because it only stores the non-zero values and their positions in memory. This is particularly useful when working with large datasets with many zero or missing values.

The default `DictVectorizer` configuration is a sparse matrix. For Week 3, Question 6, using the default sparse configuration is beneficial due to the size of the matrix. Training the model was also more performant and didn’t produce an error message like dense mode.