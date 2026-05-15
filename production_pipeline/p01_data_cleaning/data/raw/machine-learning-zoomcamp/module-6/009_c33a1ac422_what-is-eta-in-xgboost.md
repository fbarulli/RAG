---
id: c33a1ac422
question: What is eta in XGBoost
sort_order: 9
---

Sometimes someone might wonder what eta means in the tunable hyperparameters of XGBoost and how it helps the model.

ETA is the learning rate of the model. XGBoost uses gradient descent to calculate and update the model. In gradient descent, we are looking for the minimum weights that help the model to learn the data very well. This minimum weight for the features is updated each time the model passes through the features and learns them during training. Tuning the learning rate helps you tell the model what speed it would use in deriving the minimum for the weights.