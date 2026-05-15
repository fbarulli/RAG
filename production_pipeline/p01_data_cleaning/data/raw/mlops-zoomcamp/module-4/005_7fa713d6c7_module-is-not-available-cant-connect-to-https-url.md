---
id: 7fa713d6c7
question: module is not available (Can't connect to HTTPS URL)
sort_order: 5
---

First, check if the SSL module is configured with the following command:

```bash
python -m ssl
```

If the output is empty, there is no problem with the SSL configuration. Then you should upgrade your pipenv package in your current environment to resolve the problem.