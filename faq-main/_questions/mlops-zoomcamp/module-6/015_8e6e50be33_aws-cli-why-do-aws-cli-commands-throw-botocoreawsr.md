---
id: 8e6e50be33
question: 'AWS CLI: Why do AWS CLI commands throw <botocore.awsrequest.AWSRequest
  object at 0x74c89c3562d0> type messages when listing or creating AWS S3 buckets
  with LocalStack?'
sort_order: 15
---

If you encounter such messages when trying to list your AWS S3 buckets (e.g., `aws --endpoint-url=http://localhost:4566 s3 ls`), you can try configuring AWS with the same region, access key, and secret key as those in your `docker-compose` file.

To configure AWS CLI, follow these steps:

1. After installing the AWS CLI, run the following command in your terminal:
   
   ```bash
   aws configure
   ```

2. Input the required information when prompted:
   - **AWS Access Key ID:** [Example: `abc`]
   - **AWS Secret Access Key:** [Example: `xyz`]
   - **Default region name:** [Example: `eu-west-1`]