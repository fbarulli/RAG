---
id: 37e1b6b9bc
question: MLflow Autolog not working
sort_order: 14
---

Make sure `mlflow.autolog()` (or framework-specific autolog) is written **before** `with mlflow.start_run()`, not after.

Also, ensure that all dependencies for the autologger are installed, including `matplotlib`. A warning about uninstalled dependencies will be raised.