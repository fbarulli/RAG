---
id: ab0c17d01c
question: Filter a dataset by using its values
sort_order: 7
---

We can filter a dataset by using its values as shown below:

```python
# Using OR condition
 df = df[(df['ocean_proximity'] == '<1H OCEAN') | (df['ocean_proximity'] == 'INLAND')]
```

You can use `|` for 'OR', and `&` for 'AND'.

Alternative method:

```python
# Using isin()
 df = df[df['ocean_proximity'].isin(['<1H OCEAN', 'INLAND'])]
```