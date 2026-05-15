---
id: f7cd7d85ab
question: 'mlflow ui on Windows FileNotFoundError: [WinError 2] The system cannot
  find the file specified'
sort_order: 28
---

**Problem:** After successfully installing `mlflow` using `pip install mlflow` on a Windows system, running the `mlflow ui` command results in the error:

```plaintext
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**Solution:**

Add `C:\Users\{User_Name}\AppData\Roaming\Python\Python39\Scripts` to the `PATH`.