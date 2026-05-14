---
id: e00d5b3f65
question: Permission denied (publickey) Error (when you remove your public key on
  the AWS machine)
sort_order: 43
---

If you encounter a "Permission denied (publickey)" error after removing your public key from an AWS machine, follow these steps:

1. Access your machine via Session Manager to recreate your public key. Refer to the guide for more details: [Fix Permission Denied Errors](https://repost.aws/knowledge-center/ec2-linux-fix-permission-denied-errors).

2. To retrieve your old public key, use this command:
   
   ```bash
   ssh-keygen -y -f /path_to_key_pair/my-key-pair.pem
   ```

   Replace `/path_to_key_pair/my-key-pair.pem` with the actual path to your key pair.

3. For additional instructions on retrieving the public key, consult the AWS documentation: [Retrieving the Public Key](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/describe-keys.html#retrieving-the-public-key).