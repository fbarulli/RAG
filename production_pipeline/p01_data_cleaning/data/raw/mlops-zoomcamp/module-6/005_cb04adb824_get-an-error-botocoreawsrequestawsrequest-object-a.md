---
id: cb04adb824
question: Get an error "<botocore.awsrequest.AWSRequest object at 0x7fbaf2666280>"
  after running an AWS CLI command
sort_order: 5
---

When executing an AWS CLI command (e.g., `aws s3 ls`), you may encounter the error:

```plaintext
<botocore.awsrequest.AWSRequest object at 0x7fbaf2666280>
```

To fix this, set the AWS CLI environment variables:

```bash
export AWS_DEFAULT_REGION=eu-west-1
export AWS_ACCESS_KEY_ID=foobar
export AWS_SECRET_ACCESS_KEY=foobar
```

Their values are not important; any values will suffice.