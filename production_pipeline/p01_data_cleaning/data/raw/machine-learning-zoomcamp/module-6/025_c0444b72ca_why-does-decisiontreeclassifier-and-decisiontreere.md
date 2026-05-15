---
id: c0444b72ca
question: Why does DecisionTreeClassifier and DecisionTreeRegressor not throw an error
  when there are nan (missing) values in the feature matrix?
sort_order: 25
---

In lesson 6.3 around 6:00, there is an error due to missing values. Subsequently, `.fillna(0)` is used on `df_train` to deal with this. However, since version 1.3, support for missing values has been added for `DecisionTreeClassifier` and `DecisionTreeRegressor`.

More details can be found [here](https://scikit-learn.org/1.5/whats_new/v1.3.html) under `sklearn.tree`. 
