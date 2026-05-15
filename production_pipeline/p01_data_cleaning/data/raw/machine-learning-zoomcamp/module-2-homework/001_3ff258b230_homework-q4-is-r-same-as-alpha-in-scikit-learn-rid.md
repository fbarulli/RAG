---
id: 3ff258b230
question: 'Homework Q4: Is r same as alpha in Scikit-Learn Ridge?'
sort_order: 1
---

In the context of regression, particularly with regularization:

- `r` typically represents the regularization parameter in some algorithms. It controls the strength of the penalty applied to the coefficients of the regression model to prevent overfitting.

- In `sklearn.Ridge()`, the parameter `alpha` serves the same purpose as `r`. It specifies the amount of regularization applied to the model. A higher value of `alpha` increases the amount of regularization, which can reduce model complexity and improve generalization.

`r` and `alpha` are both regularization parameters and control the "strength" of regularization. Increasing these values will lead to stronger regularization. However, the mathematical implementation differs:

- **`sklearn.Ridge()`**:

  ```
  ||y - Xw||^2_2 + alpha * ||w||^2_2
  ```

- **Lesson's Notebook (`train_linear_regression_reg` function)**:
  
  ```python
  XTX = XTX + r * np.eye(XTX.shape[0])
  ```
  
  `r` adds “noise” to the main diagonal to prevent multicollinearity, which “breaks” finding the inverse matrix.

For further reference, see the [sklearn.Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) and the [lesson’s notebook](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/02-regression/notebook.ipynb).