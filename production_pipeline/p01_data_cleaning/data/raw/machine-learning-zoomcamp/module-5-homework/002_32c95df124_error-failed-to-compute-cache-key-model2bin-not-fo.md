---
id: 32c95df124
question: 'Error: failed to compute cache key: "/model2.bin" not found: not found'
sort_order: 2
---

Initially, I did not assume there was a `model2`. I copied the original `model1.bin` and `dv.bin`. Then when I tried to load using:

```dockerfile
COPY ["model2.bin", "dv.bin", "./"]
```

I got the error above in MINGW64 (git bash) on Windows.

The temporary solution I found was to use:

```dockerfile
COPY ["*", "./"]
```

This seems to combine all the files from the original Docker image and the files in your working directory.