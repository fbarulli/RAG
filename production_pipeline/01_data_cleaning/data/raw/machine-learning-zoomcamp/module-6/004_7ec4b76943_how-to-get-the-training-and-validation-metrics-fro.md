---
id: 7ec4b76943
images:
- description: 'image #1'
  id: image_1
  path: images/machine-learning-zoomcamp/image_ca7081bf.png
question: How to get the training and validation metrics from XGBoost?
sort_order: 4
---

During the XGBoost lesson, we created a parser to extract the training and validation AUC from the standard output. However, we can accomplish that in a more straightforward way.

We can use the `evals_result` parameter, which takes an empty dictionary and updates it for each tree. Additionally, you can store the data in a dataframe and plot it in an easier manner.

<{IMAGE:image_1}>
