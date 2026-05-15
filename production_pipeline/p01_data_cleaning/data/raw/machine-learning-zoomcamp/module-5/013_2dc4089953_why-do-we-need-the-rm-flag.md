---
id: 2dc4089953
question: Why do we need the --rm flag?
sort_order: 13
---

When running Docker containers, using the `--rm` flag ensures that the containers are removed upon exit. This helps in managing disk space by preventing the accumulation of stopped containers, which can consume unnecessary space.

Here are the main points regarding the use of the `--rm` flag:

- **Space Management**: Running containers with the `--rm` flag prevents the accumulation of stopped containers, thus conserving disk space.
- **Development and Testing**: During these phases, containers often don't need to persist, making the `--rm` flag useful for automatic removal.
- **Images vs Containers**: It's crucial to differentiate between them. Images are not modified upon execution; containers are the instances created from these images. The `--rm` flag affects containers, not the images themselves.
- **Rebuilding**: When a file like a Pipfile changes, the image is rebuilt, often under the same or a new tag, and the `--rm` flag helps maintain a clean environment.

Use `docker images` to list images and `docker ps -a` to list all containers, helping you manage your Docker resources efficiently.