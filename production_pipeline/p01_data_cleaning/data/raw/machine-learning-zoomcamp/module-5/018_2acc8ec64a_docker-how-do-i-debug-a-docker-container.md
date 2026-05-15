---
id: 2acc8ec64a
question: 'Docker: How do I debug a docker container?'
sort_order: 18
---

To debug a Docker container, follow these steps:

1. **Launch the container image in interactive mode** while overriding the entrypoint, so that it starts with a bash command:
   
   ```bash
   docker run -it --entrypoint bash <image>
   ```

2. If the container is already running, **execute a command in the specific container**:

   - First, find the container ID by listing the running containers:
     
     ```bash
     docker ps
     ```
   
   - Then, execute bash in the container:
     
     ```bash
     docker exec -it <container-id> bash
     ```