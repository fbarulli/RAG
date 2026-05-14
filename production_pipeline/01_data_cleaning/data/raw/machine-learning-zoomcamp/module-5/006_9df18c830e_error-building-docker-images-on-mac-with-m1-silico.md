---
id: 9df18c830e
question: Error building Docker images on Mac with M1 silicon
sort_order: 6
---

Do you get errors building the Docker image on the Mac M1 chipset?

The error received was:

```
Could not open '/lib64/ld-linux-x86-64.so.2': No such file or directory
```

To fix this error:

1. Open the `mlbookcamp-code/course-zoomcamp/01-intro/environment/Dockerfile`.

2. Replace line 1 with:
   
   ```dockerfile
   FROM --platform=linux/amd64 ubuntu:latest
   ```

3. Now build the image as specified.

Note: Building the image may take over 2 hours, but it should complete successfully.