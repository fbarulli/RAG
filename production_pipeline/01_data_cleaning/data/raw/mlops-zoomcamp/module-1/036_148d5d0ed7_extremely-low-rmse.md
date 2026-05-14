---
id: 148d5d0ed7
question: Extremely low RMSE
sort_order: 36
---

**Problem:** I’m facing an extremely low RMSE score (e.g., 4.3451e-6) - what should I do?

**Answer:**

- Recheck your code to see if your model is inadvertently learning the target before making predictions.
- Ensure that the target variable is not included as a parameter while fitting the model. Including it can result in misleadingly low scores.
- Verify that `X_train` does not contain any part of your `y_train`. This applies to the validation set as well.
- Adjust your data handling to avoid data leakage between your features and the target.