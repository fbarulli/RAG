---
id: fee2e5fbda
question: Visualize Feature Importance by using horizontal bar chart
sort_order: 18
---

To make it easier to determine which features are important, we can use a horizontal bar chart to illustrate feature importance sorted by value.

1. **Extract the feature importances from the model**

    ```python
    feature_importances = list(zip(features_names, rdr_model.feature_importances_))
    importance_df = pd.DataFrame(feature_importances, columns=['feature_names', 'feature_importances'])
    ```

2. **Sort the DataFrame in descending order using the feature_importances value**

    ```python
    importance_df = importance_df.sort_values(by='feature_importances', ascending=False)
    ```

3. **Create a horizontal bar chart**

    ```python
    plt.figure(figsize=(8, 6))
    sns.barplot(x='feature_importances', y='feature_names', data=importance_df, palette='Blues_r')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.title('Feature Importance Chart')
    ```

