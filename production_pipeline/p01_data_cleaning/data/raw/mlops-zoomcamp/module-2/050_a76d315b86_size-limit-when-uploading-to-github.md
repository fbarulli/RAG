---
id: a76d315b86
question: Size limit when uploading to GitHub
sort_order: 50
---

To manage size limits effectively when uploading to GitHub, add the `mlruns` and `artifacts` directories to your `.gitignore`, like this:

```
02-experiment-tracking/mlruns
02-experiment-tracking/runnin-mflow-examples/mlruns
02-experiment-tracking/homework/mlruns
02-experiment-tracking/homework/artifacts
```