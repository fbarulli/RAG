---
id: 2383fdd2f6
question: 'Git: Created repo without .gitignore'
sort_order: 7
---

If you created a repository without a .gitignore, follow these steps to add one:

1. Open Terminal.
2. Navigate to the location of your Git repository.
3. Create a .gitignore file for your repository:

   ```bash
   touch .gitignore
   ```

4. Locate the .gitignore file. If you already have it, open it.
5. Edit the .gitignore file and add the following lines:

   ```plaintext
   # Python
   *.pyc
   __pycache__/
   *.py[cod]
   *$
   ```

6. Save the changes to the .gitignore file.
7. Commit the changes.