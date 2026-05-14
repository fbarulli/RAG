---
id: 460343cb44
question: When using Autologging, do I need to set a training parameter to track it
  on Mlflow UI?
sort_order: 47
---

No, in the official documentation it’s mentioned that autologging keeps track of the parameters even when you do not explicitly set them when calling `.fit`.

You can run the training, only setting the parameters you want, but you can check all the parameters in the MLflow UI.