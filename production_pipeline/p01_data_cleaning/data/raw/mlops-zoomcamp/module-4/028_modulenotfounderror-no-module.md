---
id: '1355481387'
question: 'ModuleNotFoundError: No module named ''pipenv.patched.pip._vendor.urllib3.response'''
sort_order: 28
---

If you're encountering the error:

```
ModuleNotFoundError: No module named 'pipenv.patched.pip._vendor.urllib3.response'
```

Follow these steps to resolve it:

1. Reinstall `pipenv` with the following command:
   
   ```bash
   pip install pipenv --force-reinstall
   ```

2. If you see an error referring to `site-packages\pipenv\patched\pip\_vendor\urllib3\connectionpool.py`, then:

   - Upgrade `pip` and install `requests`:
   
     ```bash
     pip install -U pip
     pip install requests
     ```