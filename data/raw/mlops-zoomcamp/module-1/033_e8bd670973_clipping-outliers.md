---
id: e8bd670973
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_2715561d.png
question: Clipping outliers
sort_order: 33
---

How to check that we removed the outliers?

Use the pandas function `describe()` which can provide a report of the data distribution along with the statistics to describe the data. For example, after clipping the outliers using a boolean expression, the min and max can be verified using:

```python
 df['duration'].describe()
```

<{IMAGE:image_1}>