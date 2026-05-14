---
id: c1768edac2
question: Why does MlflowClient no longer support list_experiments?
sort_order: 1
---

Older versions of MLflow used `client.list_experiments()`, but in recent versions, this method was replaced.

Use `client.search_experiments()` instead.