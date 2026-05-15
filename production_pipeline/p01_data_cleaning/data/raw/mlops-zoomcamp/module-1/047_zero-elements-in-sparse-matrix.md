---
id: '8894361030'
question: Zero elements in sparse matrix (AKA when dictionary vectorizer / categorical
  X transformation fails)
sort_order: 47
---

Seeing a message like:

```
<2855951x515 sparse matrix of type '<class 'numpy.float64'>' with 0 stored elements in Compressed Sparse Row format>
```

This issue might occur because your variables, intended for vectorization, were imported as floating point numbers rather than integers. This can lead to nonsensical models. To resolve this, convert your data with the following code (assuming `dg` is your dataframe and `categorical` stores the names of your variables to be vectorized):

```python
dg[categorical] = dg[categorical].round(0).astype(int).astype(str)
```