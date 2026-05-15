---
id: ba636cdb18
question: Getting NaNs after applying .mean()
sort_order: 4
---

I was using for loops to apply RMSE to a list of `y_val` and `y_pred`. However, the resulting RMSE was all NaN.

I discovered that the issue occurred during the mean calculation step in the RMSE function, after squaring the error. There were NaNs in the array, which I traced back to the initial data splitting step. I had only used `fillna(0)` on the training data, not on the validation and test data. 

The problem was resolved by applying `fillna(0)` to all datasets (train, validation, and test). My for loops now successfully compute RMSE for all seed values.

