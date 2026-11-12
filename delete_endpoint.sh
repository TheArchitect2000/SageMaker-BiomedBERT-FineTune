#!/bin/bash
# Stop the bill: delete the endpoint, the endpoint configuration and the model. Run in AWS CloudShell.
# The endpoint goes first: it is the only one of the three that bills by the hour.
aws sagemaker delete-endpoint        --endpoint-name        biomedbert-ner-endpoint
aws sagemaker delete-endpoint-config --endpoint-config-name biomedbert-ner-config
aws sagemaker delete-model           --model-name           biomedbert-ner-model
echo "Endpoints left (should be empty):"
aws sagemaker list-endpoints --query 'Endpoints[].EndpointName'
