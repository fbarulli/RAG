---
id: f74891d445
images:
- description: 'image #1'
  id: image_1
  path: images/mlops-zoomcamp/image_09b2050c.png
question: How can I enable communication between Docker containers when invoked from
  a Kestra task?
sort_order: 26
---

![Diagram of Docker Container Communication](<{IMAGE:image_1}>)

Use the `docker.Run` plugin in your Kestra task to run containers. This plugin supports advanced Docker options like custom networks.

For local development, you can use `networkMode: host` to allow containers to access services on your host (e.g., MLflow running on localhost).

Example:

```yaml
networkMode: host
```

Note:
- `host` mode is only supported on Linux.
- For Docker Desktop on Windows/macOS, use `host.docker.internal` or create a shared Docker network.

**Best Practice:**

In production setups, tools like MLflow should run outside Kestra and be accessed over a stable URI (e.g., a cloud endpoint or a container with a known hostname in a shared network).