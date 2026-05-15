---
id: feed303fc9
question: 'ConvergenceWarning: The max_iter was reached'
sort_order: 13
---

If you're encountering the following warning:

ConvergenceWarning: The max_iter was reached which means the coef_ did not converge

This usually happens because the solver the model uses
is sensitive to feature scales.

You can do the following to address it:


1. **Normalize Numerical Features**  
   - Scale your numerical features using techniques like `StandardScaler` or `MinMaxScaler`.  
   - This ensures that all numerical features are on a similar scale, which helps the solver converge.  

2. **Encode Categorical Features**  
   - Apply `OneHotEncoder` (OHE) to categorical features to represent them as binary vectors.  
   - Use `sparse=False` when necessary to return a dense array.  

3. **Separate and Combine Features**  
   - Process numerical and categorical features separately (scaling for numerical, OHE for categorical).  
   - Combine them afterward into a single feature matrix (`X_train`) to use as input for Ridge regression.  

4. **Experiment with Different Scalers**  
   - If issues persist, try different scalers as Ridge can behave differently depending on feature scaling.  

By following these steps, you can reduce convergence errors and improve model stability. For a detailed example, see this notebook: [notebook-scaling-ohe.ipynb](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/03-classification/notebook-scaling-ohe.ipynb).
