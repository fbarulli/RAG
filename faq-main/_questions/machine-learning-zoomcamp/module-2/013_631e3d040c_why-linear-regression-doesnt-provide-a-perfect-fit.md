---
id: 631e3d040c
images:
- description: 'image #1'
  id: image_1
  path: images/machine-learning-zoomcamp/image_008f5d2a.png
question: Why linear regression doesn’t provide a “perfect” fit?
sort_order: 13
---

Linear regression often provides a good approximation of the underlying relationship but rarely achieves a "perfect" fit in real-world applications.

**Q:** Why is `y_pred` different from `y`?

In lesson 2.8, the question arises: after training `X_train` to get the weights, shouldn't multiplying by `X_train` give exactly `y`?

**A:** Linear regression is a simple model and should not fit 100%, as this would indicate overfitting. Consider a single feature `X`:

<{IMAGE:image_1}>

As the model is linear, how would you draw a line to fit all the "dots"?

You could "fit" all the "dots" using something like `scipy.optimize.curve_fit` (non-linear least squares), but consider how it would perform on previously unseen data.

Refer to: [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)