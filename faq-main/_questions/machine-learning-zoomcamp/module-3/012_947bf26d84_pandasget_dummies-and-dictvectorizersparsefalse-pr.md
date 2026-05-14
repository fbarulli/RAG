---
id: 947bf26d84
images:
- description: 'image #1'
  id: image_1
  path: images/machine-learning-zoomcamp/image_e1899130.png
question: 'pandas.get_dummies() and DictVectorizer(sparse=False) produce the same
  type of one-hot encodings:'
sort_order: 12
---

<{IMAGE:image_1}>

`DictVectorizer(sparse=True)` produces [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) format, which is both more memory efficient and converges better during fit().

It stores non-zero values and indices instead of adding a column for each class of each feature, which can result in large numbers of columns (e.g., models of cars).

Using "sparse" format is slower (around 6-8 minutes for Q6 task - Linear/Ridge Regression) for a high number of classes (like car models) and produces slightly worse results in both Logistic and Linear/Ridge Regression.

It also generates convergence warnings for Linear/Ridge Regression.