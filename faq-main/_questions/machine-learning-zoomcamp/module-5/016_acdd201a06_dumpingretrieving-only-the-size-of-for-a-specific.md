---
id: acdd201a06
question: Dumping/Retrieving only the size of for a specific Docker image
sort_order: 16
---

To list all information for all local Docker images, you can use the following commands:

```bash
docker images
docker image ls
```

To retrieve information for a specific image, use:

```bash
docker image ls <image_name>
```

Or alternatively:

```bash
docker images <image_name>
```

To dump only the size of a specified image, use the `--format` option. This will display only the image size:

```bash
docker image ls --format "{{.Size}}" <image_name>
```

Or alternatively:

```bash
docker images --format "{{.Size}}" <image_name>
```