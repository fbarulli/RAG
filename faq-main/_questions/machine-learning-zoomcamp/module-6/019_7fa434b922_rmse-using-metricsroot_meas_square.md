---
id: 7fa434b922
question: RMSE using metrics.root_meas_square()
sort_order: 19
---

Instead of using `np.sqrt()` as a second step, you can extract it using:

```python
mean_squared_error(y_val, y_predict_val, squared=False)
```