---
id: 0f19932601
question: '[pipenv.exceptions.ResolutionFailure]: Warning: Your dependencies could
  not be resolved. You likely have a mismatch in your sub-dependencies'
sort_order: 13
---

**Problem:** If you run `pipenv install` and get this message, it may indicate a mismatch in your sub-dependencies.

**Solution:**

1. You may need to manually update `Pipfile` and `Pipfile.lock`.
2. Run the following command to resolve the dependency issues:

   ```bash
   pipenv lock
   ```