---
id: 4aa018f478
question: Activating Anaconda env in .bashrc
sort_order: 41
---

**Problem:** Installing Anaconda didn’t modify the `.bashrc` profile. This means the Anaconda environment was not activated after exiting and relaunching the Unix shell.

**Solution:**

- **For Bash:**
  - Initiate conda again, which will add entries for Anaconda in the `.bashrc` file.
  
  ```bash
  cd YOUR_PATH_ANACONDA/bin 
  ./conda init bash
  ```
  
  - This will automatically edit your `.bashrc`.

- **Reload:**
  
  ```bash
  source ~/.bashrc
  ```