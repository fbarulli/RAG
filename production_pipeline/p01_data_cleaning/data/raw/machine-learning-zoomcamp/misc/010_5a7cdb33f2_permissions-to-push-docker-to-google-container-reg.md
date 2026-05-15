---
id: 5a7cdb33f2
question: Permissions to push docker to Google Container Registry
sort_order: 10
---

When you try to push the Docker image to Google Container Registry and receive the message:

```
unauthorized: You don't have the needed permissions to perform this operation, and you may have invalid credentials.
```

Follow these steps:

1. Install the Google Cloud SDK from [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install).
2. Run the following command in your console:

   ```bash
   gcloud auth configure-docker
   ```