---
id: ced071dd97
images:
question: 'docker: build ERROR COPY [Pipfile, Pipfile.lock]'
sort_order: 25
---

```
% docker build -t zoomcamp_test .

[+] Building 0.1s (10/10) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 332B
 => [internal] load .dockerignore
 => => transferring context: 2B
 => [internal] load build context
 => => transferring context: 2B
 => [internal] load metadata for docker.io/svizor/zoomcamp-model:3.9.12-slim
 => [1/6] FROM docker.io/svizor/zoomcamp-model:3.9.12-slim
 => [internal] load build context
 => => transferring context: 2B
 => CACHED [2/6] RUN pip install pipenv
 => CACHED [3/6] WORKDIR /app
 => ERROR [4/6] COPY [Pipfile, Pipfile.lock, ./]
 => CACHED [5/6] RUN pipenv install --system --deploy
 => ERROR [6/6] COPY [q5_predict.py, model1.bin, dv.bin, ./]
```


This error occurred because I used single quotes around the filenames. Stick to double quotes.