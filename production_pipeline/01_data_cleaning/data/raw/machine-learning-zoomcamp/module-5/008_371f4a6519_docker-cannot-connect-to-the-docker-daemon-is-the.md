---
id: 371f4a6519
question: 'Docker: Cannot connect to the docker daemon. Is the Docker daemon running?'
sort_order: 8
---

Ensure Docker Daemon Is Running

**On Windows:**

- Open Docker Desktop (admin rights may be required).
- Check if it’s running, and restart Docker Desktop if necessary.

**On Linux:**

1. Run the following command to start the Docker daemon:
   
   ```bash
   sudo systemctl start docker
   ```
2. Verify it’s running with:

   ```bash
   sudo systemctl status docker
   ```

Verify Docker Group Membership (Linux Only)

- Check if your user is in the Docker group:

  ```bash
  groups $USER
  ```

- If "docker" isn’t listed, add yourself with:

  ```bash
  sudo usermod -aG docker $USER
  ```

- Log out and back in to apply changes.

Restart the Docker Service (Linux)

```bash
sudo systemctl restart docker
```

Check Docker Socket Permissions (Linux)

- Run the following command to confirm Docker socket permissions:

  ```bash
  sudo chmod 666 /var/run/docker.sock
  ```

Try Running Docker with sudo (Linux)

- Run the following to check if permissions are causing the issue:

  ```bash
  sudo docker ps
  ```

Test Docker Setup

- Run a test Docker command to verify connection:

  ```bash
  docker run hello-world
  ```

Solution for WSL Error

If you’re encountering the error on WSL, re-install Docker by removing the Docker installation from WSL and installing Docker Desktop on your host machine (Windows).

**On Linux,** start the docker daemon with either of these commands:

- Start the Docker daemon:

  ```bash
  sudo dockerd
  ```

  or

  ```bash
  sudo service docker start
  ```
