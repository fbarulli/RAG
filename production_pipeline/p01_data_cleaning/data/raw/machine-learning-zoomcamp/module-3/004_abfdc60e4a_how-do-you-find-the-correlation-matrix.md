---
id: abfdc60e4a
question: How do you find the correlation matrix?
sort_order: 4
---

First, you have to consider whether the data is numerical or categorical. If it’s numerical, you can correlate it directly. If it’s categorical, you can find the correlations indirectly by vectorizing the data using One-Hot encoding or a similar method.

To determine if data is numerical, check the `dtypes` of the DataFrame. Data types such as integer and float are numerical, while types such as objects are categorical. You can correlate the numerical data by specifying which columns are numerical and using that as input to a correlation matrix.

Example:

```python
numerical = ['tenure', 'monthlycharges', 'totalcharges']

correlation_matrix = df[numerical].corr()
print(correlation_matrix)
```