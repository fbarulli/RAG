---
id: 1bc7beaf5d
question: Install docker in WSL2 without installing Docker Desktop
sort_order: 46
---

If you want to install Docker in WSL2 on Windows without Docker Desktop, follow these steps:

1. **Install Docker**

   You can ignore the warnings during installation.
   
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   ```
   
2. **Add Your User to the Docker Group**
   
   ```bash
   sudo usermod -aG docker $USER
   ```

3. **Enable the Docker Service**
   
   ```bash
   sudo systemctl enable docker.service
   ```

4. **Test the Installation**

   Verify that both Docker and Docker Compose are installed successfully.
   
   ```bash
   docker --version
   docker compose version
   docker run hello-world
   ```

5. **Ensure Docker Starts Automatically**
   
   If the service does not start automatically after restarting WSL, update your `.profile` or `.zprofile` file with:
   
   ```bash
   if grep -q "microsoft" /proc/version > /dev/null 2>&1; then
      if service docker status 2>&1 | grep -q "is not running"; then
         wsl.exe --distribution "${WSL_DISTRO_NAME}" --user root \
         --exec /usr/sbin/service docker start > /dev/null 2>&1
      fi
   fi
   ```