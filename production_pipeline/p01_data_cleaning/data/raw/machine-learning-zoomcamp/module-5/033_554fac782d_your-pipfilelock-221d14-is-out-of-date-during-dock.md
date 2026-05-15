---
id: 554fac782d
question: Your Pipfile.lock (221d14) is out of date (during Docker build)
sort_order: 33
---

If during running the docker build command, you get an error like this:

```
Your Pipfile.lock (221d14) is out of date. Expected: (939fe0).

Usage: pipenv install [OPTIONS] [PACKAGES]...

ERROR:: Aborting deploy
```

You can try the following solutions:

1. **Delete and Rebuild Pipfile.lock:**
   - Delete the `Pipfile.lock` using the command:
   
     ```bash
     rm Pipfile.lock
     ```
   
   - Rebuild the lock file:
   
     ```bash
     pipenv lock
     ```
   
   - Retry the `docker build` command.

2. **Remove and Recreate Pipenv Environment:**
   - Remove the pipenv environment:
   
     ```bash
     pipenv --rm
     ```
   
   - Remove the Pipfile and Pipfile.lock:
   
     ```bash
     rm Pipfile*
     ```
   
   - Create a new environment before retrying the Docker build.