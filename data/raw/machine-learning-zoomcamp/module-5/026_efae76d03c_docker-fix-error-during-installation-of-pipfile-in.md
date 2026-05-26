---
id: efae76d03c
question: 'Docker: Fix error during installation of Pipfile inside Docker container'
sort_order: 26
---

```
(hw5) (base) home@ls-MacBook-Pro hw5 % docker build -t zoomcamp_test .
[+] Building 19.7s (9/10)
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 332B
 => [internal] load .dockerignore
 => => transferring context: 2B
 => [internal] load metadata for docker.io/svizor/zoomcamp-model:3.9.12-slim
 => CACHED [1/6] FROM docker.io/svizor/zoomcamp-model:3.9.12-slim
 => [internal] load build context
 => => transferring context: 19.77kB
 => [2/6] RUN pip install pipenv
 => [3/6] WORKDIR /app
 => [4/6] COPY [Pipfile, Pipfile.lock, ./]
 => ERROR [5/6] RUN pipenv install --system --deploy
------
 > [5/6] RUN pipenv install --system --deploy:
#8 0.659 Your Pipfile.lock (65dad0) is out of date. Expected: (f3760a).
#8 0.660 Usage: pipenv install [OPTIONS] [PACKAGES]...
#8 0.660 ERROR:: Aborting deploy
```

I tried the first solution on Stackoverflow which recommended running `pipenv lock` to update the Pipfile.lock.

However, this didn’t resolve it. The following switch to the pipenv installation worked:

```bash
RUN pipenv install --system --deploy --ignore-pipfile
```