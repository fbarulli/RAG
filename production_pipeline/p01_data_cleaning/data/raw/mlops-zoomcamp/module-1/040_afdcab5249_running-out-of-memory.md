---
id: afdcab5249
question: Running out of memory
sort_order: 40
---

**Problem:** The output of DictVectorizer was consuming too much memory, causing an inability to fit the linear regression model before running out of memory on a 16 GB machine.

**Solution:**

- In the example for DictVectorizer on the scikit-learn [website](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html), the parameter `sparse` is set as `False`. While this helps with viewing results, it greatly increases memory usage.
- To address this, either set `sparse=True`, or leave it at the default setting, which is also `True`. 

By using `sparse=True`, memory usage will be reduced, allowing for more efficient model fitting.