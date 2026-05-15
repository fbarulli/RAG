---
id: bcd4190972
question: '.gitignore: how-to'
sort_order: 8
---

If you create a folder `data` and download datasets or raw files to your local repository, you might want to push all your code to a remote repository without including these files or folders. To achieve this, use a `.gitignore` file.

Follow these steps to create a `.gitignore` file:

1. Create an empty `.txt` file using a text editor or command line.
2. Save as `.gitignore` (ensure you use the dot symbol).
3. Add rules:
   - `*.parquet` to ignore all Parquet files.
   - `data/` to ignore all files in the `data` folder.

For more patterns, read the [GIT documentation](https://git-scm.com/docs/gitignore).

<>{IMAGE:image_id}