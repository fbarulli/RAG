---
id: 885a92b3ee
question: How to replace distplot with histplot
sort_order: 23
---

To replace `sns.distplot` with `sns.histplot`, you can use the following syntax:

```python
sns.distplot(df_train["duration"])
```

Can be replaced with:

```python
sns.histplot(
    df_train["duration"], kde=True,
    stat="density", kde_kws=dict(cut=3), bins=50,
    alpha=.4, edgecolor=(1, 1, 1, 0.4),
)
```

This will give you an almost identical result.