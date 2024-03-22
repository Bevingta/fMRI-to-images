# Created by Andrew Bevigton
import boto3

s3 = boto3.client('s3')

bucket_name = "arn:aws:s3:::natural-scenes-dataset"
key = <your_key_here>

response = s3.get_object(Bucket=bucket_name, Key=key)

# Read data from response
dataset_content = response['Body'].read().decode('utf-8')

print(dataset_content)
