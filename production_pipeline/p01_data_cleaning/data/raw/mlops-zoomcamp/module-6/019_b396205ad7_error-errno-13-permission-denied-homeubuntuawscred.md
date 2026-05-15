---
id: b396205ad7
question: 'Error "[Errno 13] Permission denied: ''/home/ubuntu/.aws/credentials''"
  when running any aws command'
sort_order: 19
---

After installing AWS CLI v2 on Linux, you may encounter a permission error when trying to run AWS commands that require access to your credentials. For example, when running `aws configure`, you might insert the key and secret but receive a permission error.

The issue arises because the `ubuntu` user does not have permission to read or write files in the `.aws` folder, and the `credentials` and `config` files do not exist. To resolve this:

1. Navigate to the `.aws` folder, typically located at `/home/ubuntu/.aws`.

2. Create empty `credentials` and `config` files:

   ```bash
   touch credentials
   touch config
   ```

3. Modify the file permissions:

   ```bash
   sudo chmod -R 777 credentials
   sudo chmod -R 777 config
   ```

4. Run `aws configure`, modify the keys and secret, and save them to the `credentials` file. You can then execute AWS commands from your Python scripts or the command line.