---
id: 6698c6cbd4
question: wget is not recognized as an internal or external command
sort_order: 1
---

If you encounter the error "wget is not recognized as an internal or external command", you need to install it.

- **On Ubuntu, run:**
  
  ```bash
  sudo apt-get install wget
  ```

- **On Windows, you can use Chocolatey:**
  
  ```bash
  choco install wget
  ```
  
  Or download a binary [from here](http://gnuwin32.sourceforge.net/packages/wget.htm) and add it to your PATH (e.g., `C:/tools/`).

- **On Mac, use Homebrew:**
  
  ```bash
  brew install wget
  ```

Alternatively, you can use Python libraries:

- **Python `wget` library:**

  Install it first:
  
  ```bash
  pip install wget
  ```

  Then, in your Python code:
  
  ```python
  import wget

  wget.download("URL")
  ```

- **Using `pandas` to read a CSV directly from a URL:**

  ```python
  import pandas as pd
  
  url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
  
  df = pd.read_csv(url)
  ```

  Valid URL schemes include http, ftp, s3, gs, and file.

- **Bypassing HTTPS checks (if needed):**

  ```python
  import ssl
  
  ssl._create_default_https_context = ssl._create_unverified_context
  ```

- **Using Python's `urllib` for downloading files:**

  ```python
  import urllib.request
  
  url = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv"
  
  urllib.request.urlretrieve(url, "housing.csv")
  ```

  The `urlretrieve()` function allows you to download files from URLs and save them locally. It is part of the standard Python library `urllib.request`, available on all devices and platforms.