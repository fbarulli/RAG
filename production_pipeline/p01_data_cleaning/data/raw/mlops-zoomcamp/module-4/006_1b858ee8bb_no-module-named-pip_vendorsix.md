---
id: 1b858ee8bb
question: No module named 'pip._vendor.six'
sort_order: 6
---

During scikit-learn installation via the command:

```bash
pipenv install scikit-learn==1.0.2
```

The following error is raised:

```
ModuleNotFoundError: No module named 'pip._vendor.six'
```

To resolve this issue, follow these steps:

1. Install the `python-six` package:
   
   ```bash
   sudo apt install python-six
   ```

2. Remove the existing Pipenv environment:
   
   ```bash
   pipenv --rm
   ```

3. Reinstall `scikit-learn`:

   ```bash
   pipenv install scikit-learn==1.0.2
   ```
