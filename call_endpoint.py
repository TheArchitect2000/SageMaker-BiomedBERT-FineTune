"""Send a sentence to the BiomedBERT endpoint and print the labels.

Run in AWS CloudShell (boto3 is already installed):
    python3 call_endpoint.py
    python3 call_endpoint.py "For buccal use, Levothyroxine was formulated as capsules by roller compaction."
"""
import json
import sys

import boto3

ENDPOINT = "biomedbert-ner-endpoint"
sentence = sys.argv[1] if len(sys.argv) > 1 else "Ibuprofen 200 mg tablets contain microcrystalline cellulose."

rt = boto3.client("sagemaker-runtime", region_name="us-east-1")
resp = rt.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps({"inputs": sentence}))

print(sentence)
for p in json.loads(resp["Body"].read()):
    print(f'  {p["word"]:<20} {p["entity"]:<16} {p["score"]:.2f}')
