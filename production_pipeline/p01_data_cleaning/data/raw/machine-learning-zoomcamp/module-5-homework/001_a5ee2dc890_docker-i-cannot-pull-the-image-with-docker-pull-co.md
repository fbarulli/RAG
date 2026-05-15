---
id: a5ee2dc890
question: 'Docker: I cannot pull the image with docker pull command'
sort_order: 1
---

**Problem:** When trying to pull the image using the `docker pull svizor/zoomcamp-model` command, an error occurs:

```bash
Using default tag: latest
Error response from daemon: manifest for svizor/zoomcamp-model:latest not found: manifest unknown: manifest unknown
```

**Solution:** Docker defaults to the `latest` tag. To resolve this, use the correct tag from the image description. Use the following command:

```bash
docker pull svizor/zoomcamp-model:3.10.12-slim
```