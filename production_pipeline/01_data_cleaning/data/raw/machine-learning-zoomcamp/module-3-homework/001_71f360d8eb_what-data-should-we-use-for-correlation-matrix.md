---
id: 71f360d8eb
question: What data should we use for correlation matrix?
sort_order: 1
---

Q2 asks about the correlation matrix and converting `median_house_value` from numeric to binary. Just to clarify, we are only dealing with `df_train`, not `df_train_full`, correct? The question explicitly mentions the train dataset.

Yes, it is only on `df_train`. The reason is that `df_train_full` also contains the validation dataset. At this stage, we don't want to make conclusions based on the validation data, since we want to test how we did without using that portion of the data.