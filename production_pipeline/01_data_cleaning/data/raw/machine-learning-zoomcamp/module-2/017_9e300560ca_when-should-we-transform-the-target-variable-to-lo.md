---
id: 9e300560ca
question: When should we transform the target variable to logarithm distribution?
sort_order: 17
---

When the target variable has a long tail distribution, such as prices with a wide range, you can transform it using the `np.log1p()` method. However, be aware that this method will not work if your target variable contains negative values.