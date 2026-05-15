---
id: a654b0e217
question: How to avoid accidentally pushing CSV files
sort_order: 1
---

To avoid accidentally pushing CSV files (or any specific file type) to a Git repository, you can use a `.gitignore` file.

- Add a rule to ignore CSV files by including:
  
  ```
  *.csv
  ```

- If the CSV files have already been committed, you can remove them from Git tracking but keep them locally by using the command:

  ```bash
  git rm --cached filename.csv
  ```