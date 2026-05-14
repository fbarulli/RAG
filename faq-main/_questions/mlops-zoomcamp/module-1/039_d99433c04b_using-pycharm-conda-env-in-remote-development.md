---
id: d99433c04b
question: Using PyCharm & Conda env in remote development
sort_order: 39
---

**Problem:** PyCharm (remote) doesn’t see the conda execution path, preventing the use of a conda environment located on a remote server.

**Solution:**

1. On the remote server's command line, run:
   
   ```bash
   conda activate envname
   ```
   
2. Then, execute:
   
   ```bash
   which python
   ```
   
   This will provide the Python execution path.
   
3. Use this path to add a new interpreter in PyCharm:
   
   - Add local interpreter.
   - Select system interpreter.
   - Enter the path obtained from the previous step.