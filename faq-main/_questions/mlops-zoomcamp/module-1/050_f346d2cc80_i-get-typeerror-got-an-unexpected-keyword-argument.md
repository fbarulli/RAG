---
id: f346d2cc80
question: 'I get `TypeError: got an unexpected keyword argument ''squared''` when
  using `mean_squared_error(..., squared=False)`. Why?'
sort_order: 50
---

The `squared` parameter was added in scikit-learn 0.22. In earlier versions, it is not recognized, which causes the `TypeError`.

To compute RMSE in older versions:

- Use `np.sqrt(mean_squared_error(...))`.

In scikit-learn 1.0 and later, you can use:

```python
from sklearn.metrics import root_mean_squared_error as rmse

rmse_value = root_mean_squared_error(y_train, y_pred)

print('RMSE:', rmse_value)
```

This approach is more explicit and convenient.