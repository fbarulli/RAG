---
id: e04431f782
question: Unprotected private key file!
sort_order: 14
---

After running the command:

```bash
ssh -i ~/.ssh/razer.pem ubuntu@XX.XX.XX.XX
```

I encountered the error: "unprotected private key file". To resolve this issue, ensure the file permissions are correctly set by running the following command:

```bash
chmod 400 ~/.ssh/razer.pem
```

For more detailed steps, see this [guide](https://99robots.com/how-to-fix-permission-error-ssh-amazon-ec2-instance/).