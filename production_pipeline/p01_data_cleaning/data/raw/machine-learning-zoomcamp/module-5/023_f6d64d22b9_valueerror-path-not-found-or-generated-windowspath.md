---
id: f6d64d22b9
question: 'ValueError: Path not found or generated: WindowsPath(''C:/Users/username/.virtualenvs/envname/Scripts'')'
sort_order: 23
---

After entering `pipenv shell`, ensure you use `exit` before `pipenv --rm`. Failing to do so may cause installation errors, making it unclear whether you are "in the shell" on Windows, as there are no clear markers for it.

If this messes up your PATH, use these terminal commands to fix it:

- **For Windows**:
  
  ```bash
  set VIRTUAL_ENV ""
  ```

- **For Unix**:
  
  ```bash
  export VIRTUAL_ENV=""
  ```

Additionally, manually re-creating the removed folder at `C:\Users\username\.virtualenvs\removed-envname` can help. The `removed-envname` can be identified in the error message.