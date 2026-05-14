---
id: b2781bc5f8
question: Running "pipenv install sklearn==1.0.2" gives errors. What should I do?
sort_order: 10
---

When installing sklearn version 1.0.2, you may encounter errors. This issue is due to the package name. Instead of "sklearn," you should use its full name. Here's how you can resolve this:

1. Use the following command to install the correct version:
   
   ```bash
   pipenv install scikit-learn==1.0.2
   ```

2. If your homework requires version 1.3.1, use the following command:
   
   ```bash
   pipenv install scikit-learn==1.3.1
   ```

Using the correct full package name should resolve the installation issues.