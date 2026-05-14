---
id: ff0f22df21
question: 'AWS CLI: ''Invalid base64'' error after running `aws kinesis put-record`'
sort_order: 9
---

**Problem Description:** 
You might encounter an 'Invalid base64' error after executing the `aws kinesis put-record` command on your local machine. This issue can occur if you are using AWS CLI version 2. In a referenced video (4.4, around 57:42), a warning is visible as the instructor is using version 1 of the CLI.

**Solution:**
To resolve this issue, use the argument `--cli-binary-format raw-in-base64-out` when executing the command. This option will encode your data string into base64 before transmitting it to Kinesis.

```bash
aws kinesis put-record --cli-binary-format raw-in-base64-out --other-parameters
```