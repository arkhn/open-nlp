# T5-BIGBIO

This repository is a toolkit for experiments using T5 on biomedical information extraction. It provides scripts to finetune a t5-based model using InstructionNER format and a simple server using FastAPI for inference.

## Usage

### Inference

To run the inference server, run the following command:

```bash
poetry run uvicorn app:app --port 8080
```

Example request:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8080/extract_entities' \
  -H 'accept: text/plain' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The data presented in this study clearly show that I-DOX binds specifically to TTR amyloid fibrils in tissues from patients with FAP."
}'
```

Reponse body:

```
- "I - DOX"
- "TTR amyloid fibrils"
- "tissues"
- "patients"
- "FAP"
```
