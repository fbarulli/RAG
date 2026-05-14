---
id: 1b89887a15
question: Homework Q1 is not clear to me. What do I do here?
sort_order: 6
---

Q1 is not making sense to me. The score should be between 0 to 1. I tried computing `roc_curve (df_train['age'], y)` and the graph does not have the model line. Can anyone clarify?'

The idea of the question is to evaluate the importance of features with respect to the prediction of the binary target variable (yes/no).

In my case, I did the following:

1. Identified the numerical features in the dataset.
2. For each feature in the list of numerical features, I calculated the AUC:
   
   ```python
   roc_auc_score(y_target, feature_vector)
   ```
   
   Here, `y_target` is the target variable, and `feature_vector` contains the values for each numerical column in the train dataset.

3. Created a data frame with two columns: the name of the numerical feature and the ROC AUC score.
4. Sorted the data frame by the ROC AUC score to determine the numerical feature with the highest ROC AUC.