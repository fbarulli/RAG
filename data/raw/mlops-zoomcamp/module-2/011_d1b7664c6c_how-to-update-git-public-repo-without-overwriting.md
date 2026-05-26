---
id: d1b7664c6c
question: How to Update Git Public Repo Without Overwriting Changes
sort_order: 11
---

**Problem:** I cloned the public repo, made edits, committed, and pushed them to my own repo. Now I want to get the recent commits from the public repo without overwriting my own changes to my own repo. Which command(s) should I use?

Below is the Git configuration:

```
[core]
  repositoryformatversion = 0
  filemode = true
  bare = false
  logallrefupdates = true
  ignorecase = true
  precomposeunicode = true

[remote "origin"]
  url = git@github.com:my_username/mlops-zoomcamp.git
  fetch = +refs/heads/*:refs/remotes/origin/*

[branch "main"]
  remote = origin
  merge = refs/heads/main
```

**Solution:**

- Fork the original repository from DataClubsTak instead of cloning it directly.
- On GitHub, navigate to your forked repository.
- Click “Fetch and Merge” under the “Fetch upstream” menu on the main page of your own repository.