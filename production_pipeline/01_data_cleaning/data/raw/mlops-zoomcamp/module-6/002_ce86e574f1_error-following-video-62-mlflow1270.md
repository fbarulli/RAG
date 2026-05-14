---
id: ce86e574f1
question: 'Error following video 6.2: mlflow=1.27.0'
sort_order: 2
---

When following the video instructions and running the Dockerfile, I encountered an error that the Dockerfile build failed on line 8 due to no matching distribution for `mlflow==1.27.0`. Below is the code output:

```bash
4.900 ERROR: No matching distribution found for mlflow==1.27.0

4.901 ERROR: Couldn't install package: {}

4.901  Package installation failed...

------

Dockerfile:8

--------------------

6 |     COPY [ "Pipfile", "Pipfile.lock", "./" ]

7 |

8 | >>> RUN pipenv install --system --deploy

9 |

10 |     COPY [ "lambda_function.py", "model.py", "./" ]

--------------------

ERROR: failed to solve: process "/bin/sh -c pipenv install --system --deploy" did not complete successfully: exit code: 1
```