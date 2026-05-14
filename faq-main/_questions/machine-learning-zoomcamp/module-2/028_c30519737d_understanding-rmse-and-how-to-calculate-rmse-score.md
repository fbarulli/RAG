---
id: c30519737d
question: Understanding RMSE and how to calculate RMSE score
sort_order: 28
---

The Root Mean Squared Error (RMSE) is one of the primary metrics to evaluate the performance of a regression model. It calculates the average deviation between the model's predicted values and the actual observed values, offering insight into the model's ability to accurately forecast the target variable. To calculate the RMSE score:

1. Import the necessary libraries:
   
   ```python
   import numpy as np
   from sklearn.metrics import mean_squared_error
   ```

2. Calculate the Mean Squared Error (MSE):
   
   ```python
   mse = mean_squared_error(actual_values, predicted_values)
   ```

3. Compute the RMSE:
   
   ```python
   rmse = np.sqrt(mse)
   print("Root Mean Squared Error (RMSE):", rmse)
   ```