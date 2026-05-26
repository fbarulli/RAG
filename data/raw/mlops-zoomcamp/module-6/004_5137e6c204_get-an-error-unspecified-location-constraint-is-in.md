---
id: 5137e6c204
question: Get an error ‘unspecified location constraint is incompatible’
sort_order: 4
---

You may get an error while creating a bucket with LocalStack and the Boto3 client:

```python
botocore.exceptions.ClientError: An error occurred (IllegalLocationConstraintException) when calling the CreateBucket operation: The unspecified location constraint is incompatible for the region specific endpoint this request was sent to.
```

To fix this, instead of creating a bucket via:

```python
s3_client.create_bucket(Bucket='nyc-duration')
```

Create it with:

```python
s3_client.create_bucket(Bucket='nyc-duration', CreateBucketConfiguration={'LocationConstraint': AWS_DEFAULT_REGION})
```