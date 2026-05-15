---
id: 3b76daefb2
question: 'Overfitting: Absurdly high RMSE on the validation dataset'
sort_order: 44
---

**Problem:** The February dataset has been used as a validation/test dataset and stripped of the outliers in a similar manner to the train dataset (taking only the rows for the duration between 1 and 60, inclusive). The RMSE obtained afterward is in the thousands.

**Solution:**

- Ensure that the sparse matrix result from `DictVectorizer` is not turned into an `ndarray`. After removing that part of the code, a correct result was achieved.

<{IMAGE:image_id}>

If there are further issues, carefully review each preprocessing step to ensure consistency between training and validation datasets.