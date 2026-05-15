---
id: '8351543816'
question: 'Homework: Why do I have different values of accuracy than the options in
  the homework?'
sort_order: 4
---

One main reason behind this issue is the method of splitting the data. For example, if we want to split the data into train/validation/test with the ratios 60%/20%/20%, different methods may yield different results even if the final ratios are the same.

1. Method 1:
   
   ```python
   df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
   df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
   ```

2. Method 2:
   
   ```python
   df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
   df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
   ```

While both methods achieve the same ratio, the data split differently, resulting in variations in accuracy. It is recommended to use the second method, as it is more consistent with the lessons and homeworks.