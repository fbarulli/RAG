---
id: 5bf804473e
question: 'Docker: Dockerizing LightGBM'
sort_order: 20
---

**Problem Description:**

```plaintext
lib_lightgbm.so Reason: image not found
```

**Solution:**

- Add the following command to your Dockerfile:
  
  ```bash
  RUN apt-get install libgomp1
  ```
  
- Modify the installer command based on your OS if needed.