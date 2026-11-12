# AWS SageMaker AI - BiomedBERT Model Fine-Tune

Fine-tune BiomedBERT on Amazon SageMaker to label drug text (drug, excipient, dosage form, strength, route, process), then deploy it to an endpoint and send it requests.

## Files

| File | What it is |
|---|---|
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | The labelled data |
| `train.py`, `requirements.txt` | The training code |
| `sourcedir.tar.gz` | `train.py` + `requirements.txt` zipped for the training job |
| `call_endpoint.py` | Sends a sentence to the endpoint and prints the labels |
| `delete_endpoint.sh` | Deletes the endpoint, its configuration and the model, so the bill stops |
| `SageMaker_FineTune_BiomedBERT.pdf` | The slides |

## Video

The full walkthrough video (`SageMaker_FineTune_BiomedBERT.mp4`, 42 min) is too big for the repo. Download it from the [Releases page](https://github.com/TheArchitect2000/SageMaker-BiomedBERT-FineTune/releases/latest).

## Try it

1. Create an S3 bucket with "sagemaker" in its name and upload `train.jsonl`, `validation.jsonl`, `test.jsonl` and `sourcedir.tar.gz`.
2. Create a SageMaker training job with the Hugging Face PyTorch training image (steps in the slides).
3. Create a model, an endpoint configuration and an endpoint from `model.tar.gz`.
4. Run `python3 call_endpoint.py` in CloudShell.
5. Run `./delete_endpoint.sh` when you're done. The endpoint bills every hour until you delete it.

Region used: us-east-1.
