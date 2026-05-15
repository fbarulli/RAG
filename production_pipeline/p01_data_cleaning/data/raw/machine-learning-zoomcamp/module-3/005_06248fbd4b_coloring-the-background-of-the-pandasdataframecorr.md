---
id: 06248fbd4b
images:
- description: 'image #1'
  id: image_1
  path: images/machine-learning-zoomcamp/image_db91a31a.png
question: Coloring the background of the pandas.DataFrame.corr correlation matrix
  directly
sort_order: 5
---

The background of any DataFrame, including the correlation matrix, can be colored based on its numerical values using the method [pandas.io.formats.style.Styler.background_gradient](https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html).

Here is an example of how to color the correlation matrix. A color map of choice can be passed; here, 'viridis' is used:

- Ensure the DataFrame contains only numerical values before calling `corr`:

```python
corr_mat = df_numerical_only.corr()

corr_mat.style.background_gradient(cmap='viridis')
```

- Here is an example of how the coloring will look using a DataFrame containing random values and applying `background_gradient` to it:

```python
np.random.seed = 3

df_random = pd.DataFrame(data=np.random.random(3*3).reshape(3,3))

print(df_random.style.background_gradient(cmap='viridis'))
```

<{IMAGE:image_1}>
