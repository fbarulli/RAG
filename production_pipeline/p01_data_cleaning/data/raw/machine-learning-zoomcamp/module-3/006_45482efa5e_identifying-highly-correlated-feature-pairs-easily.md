---
id: 45482efa5e
images:
- description: 'image #2'
  id: image_2
  path: images/machine-learning-zoomcamp/image_6476267e.png
question: Identifying highly correlated feature pairs easily through unstack
sort_order: 6
---

To identify highly correlated feature pairs using unstack:

```python
import pandas as pd

data_corr = pd.DataFrame(data_num.corr().round(3).abs().unstack().sort_values(ascending=False))

print(data_corr.head(10))
```

You can also use seaborn to create a heatmap with the correlation:

```python
import seaborn as sns

sns.heatmap(
    df[numerical_features].corr(),
    annot=True,
    square=True,
    fmt=".2g",
    cmap="crest"
)
```

To refine your heatmap and plot only a triangle, with a blue to red color gradient, that will show every correlation between your numerical variables without redundant information:

```python
# Set figure size: modify it here or create new function arguments
plt.figure(figsize=(12, 6))

# define the mask to set the values in the upper triangle to True
mask = np.triu(np.ones_like(dataframe.corr(numeric_only=True), dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

heatmap = sns.heatmap(
    dataframe.corr(numeric_only=True),
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    annot=True,
    linewidths=0.5,
)

heatmap.set_title(title, fontdict={"fontsize": 18}, pad=16)
plt.show()
```

Which outputs, in the case of a churn dataset:

<{IMAGE:image_2}>
