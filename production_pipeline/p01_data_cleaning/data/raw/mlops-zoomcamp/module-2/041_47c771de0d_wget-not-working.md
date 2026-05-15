---
id: 47c771de0d
question: wget not working
sort_order: 41
---

**Problem**

Using the `wget` command to download either data or Python scripts on Windows, I am using the notebook provided by Visual Studio and despite having a Python virtual environment, it did not recognize the `pip` command.

**Solution**

- Use `python -m pip`, this applies to any other command as well, e.g., `python -m wget`. 