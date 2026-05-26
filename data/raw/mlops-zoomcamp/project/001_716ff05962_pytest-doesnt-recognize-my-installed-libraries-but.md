---
id: 716ff05962
question: pytest doesn't recognize my installed libraries, but the script works in
  the terminal. Why?
sort_order: 1
---

This usually happens because VS Code is using a different Python interpreter than the one in your terminal. As a result, pytest can't see the packages installed in your virtual environment.

How to fix:

1. In your terminal, run:
   
   ```bash
   which python
   ```

2. In VS Code, open the command palette (Ctrl+Shift+P) and select:
   
   `Python: Select Interpreter`

3. Choose the same interpreter shown in step 1.