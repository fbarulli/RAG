---
id: '9802179265'
question: Failed to read Dockerfile
sort_order: 14
---

When you create the Dockerfile, ensure the name is `Dockerfile` without any extensions. A common mistake is naming it with an extension, such as `Dockerfile.dockerfile`, which results in an error during the image build. To avoid this, create the file simply as `Dockerfile`.

```bash
# Incorrect way:
Dockerfile.dockerfile

# Correct way:
Dockerfile
```