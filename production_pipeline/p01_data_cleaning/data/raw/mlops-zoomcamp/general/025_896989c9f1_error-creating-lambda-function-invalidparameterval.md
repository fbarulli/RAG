---
id: 896989c9f1
question: 'Error: creating Lambda Function (...): InvalidParameterValueException:
  The image manifest, config or layer media type for the source image ... is not supported.'
sort_order: 25
---

This error occurs when the Docker image you are using is a manifest list (multi-platform). AWS Lambda does not support manifest lists—it only accepts single-platform images with a standard image manifest.

**Quick fix:** Build your Docker image using `docker buildx` and specify the platform explicitly.

```bash
docker buildx build --platform linux/amd64 -t your-ecr-image:latest -f Dockerfile .
```

This ensures the image is compatible with AWS Lambda. Also, make sure that you push your image using the `--platform` option.