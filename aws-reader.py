# Created by Andrew Bevigton
import boto3

s3 = boto3.client('s3',
                  aws_access_key_id="AKIAU6GD33QB7OV4QNEO",
                  aws_secret_access_key="LXkc3gzvvB1PwljEz0YhI9kJgh/mIsiViarFxIx",
                  region_name="us-east-2")

bucket_name = "natural-scenes-dataset"
key="natural-scenes-dataset.zip"

response = s3.get_object(Bucket=bucket_name, Key=key)

# Read data from response
dataset_content = response['Body'].read().decode('utf-8')

print(dataset_content)
