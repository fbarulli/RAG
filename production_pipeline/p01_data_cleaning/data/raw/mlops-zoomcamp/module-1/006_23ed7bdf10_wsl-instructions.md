---
id: 23ed7bdf10
question: 'WSL: instructions'
sort_order: 6
---

If you wish to use WSL on your Windows machine, here are the setup instructions:

1. Install wget:
   
   ```bash
   sudo apt install wget
   ```

2. Download Anaconda from the [Anaconda download page](https://www.anaconda.com/download#downloads) using the `wget` command:
   
   ```bash
   wget <download-address>
   ```

3. Turn on Docker Desktop WSL 2:
   
   [Follow the instructions here](https://docs.docker.com/desktop/windows/wsl/#turn-on-docker-desktop-wsl-2).

4. Clone the desired GitHub repository:
   
   ```bash
   git clone <github-repository-address>
   ```

5. Install Jupyter:
   
   ```bash
   pip3 install jupyter
   ```

6. Consider using Anaconda, which includes tools like PyCharm and Jupyter.

7. Alternatively, download Miniforge for a lightweight, open-source version of conda that supports mamba for improved environment solving speed. The Texas Tech University High Performance Computing Center provides a detailed guide:
   
   [Installing Miniforge3 Guide by TTU HPCC](https://www.depts.ttu.edu/hpcc/userguides/application_guides/Miniforge.php)

8. For Windows, install WSL via:
    
    ```bash
    wsl --install
    ```

9. If Python shows as version 3.10 after installing Anaconda with Python 3.9, execute:
    
    ```bash
    source .bashrc
    ```
    
    If the issue persists, add the following to your PATH:
    
    ```bash
    export PATH="<anaconda-install-path>/bin:$PATH"
    ```

For using VSCode with WSL, refer to [VSCode on WSL](https://code.visualstudio.com/docs/remote/wsl).