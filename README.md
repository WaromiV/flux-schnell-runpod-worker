# FLUX Schnell RunPod Worker

Queue-based RunPod Serverless worker for FLUX.1-schnell image-to-image generation.

This repo uses the standard RunPod `def handler(job)` pattern instead of a custom FastAPI server. It is ready for Docker Registry deploys or GitHub-based RunPod deploys.

## Repo Layout

```text
.
├── app/
│   ├── __init__.py
│   └── logic.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── handler.py
├── README.md
├── requirements.txt
└── test_input.json
```

## What It Does

- Loads `black-forest-labs/FLUX.1-schnell` with `FluxImg2ImgPipeline`
- Accepts a RunPod job payload under `job["input"]`
- Supports `image_base64`, `image_url`, or `image_path` as the source image
- Returns the generated image as base64 in JSON
- Caches models under `/runpod-volume/huggingface` when a network volume is mounted

## Input Shape

```json
{
  "input": {
    "prompt": "make it inferno",
    "image_base64": "<raw base64 or data URI>",
    "steps": 4,
    "strength": 0.95,
    "guidance_scale": 0.0,
    "seed": 42,
    "width": 1024,
    "height": 1024,
    "output_format": "png"
  }
}
```

## Output Shape

```json
{
  "ok": true,
  "model_id": "black-forest-labs/FLUX.1-schnell",
  "offload_mode": "model",
  "seed": 42,
  "steps": 4,
  "strength": 0.95,
  "guidance_scale": 0.0,
  "width": 1024,
  "height": 1024,
  "mime_type": "image/png",
  "image_base64": "<base64>"
}
```

## Environment Variables

- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`: required for the gated model
- `MODEL_ID`: defaults to `black-forest-labs/FLUX.1-schnell`
- `OFFLOAD_MODE`: `none`, `model`, or `sequential`; default is `model`
- `PRELOAD_MODEL`: `1` loads the model at worker start; `0` delays until first job
- `MODEL_CACHE_DIR`: optional override for model cache location
- `MAX_SEQUENCE_LENGTH`: defaults to `256`

## Local Testing

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 torch
pip install -r requirements.txt
```

Set a Hugging Face token first:

```bash
export HF_TOKEN=hf_your_token_here
```

Run a local JSON test:

```bash
python handler.py
```

Or pass inline JSON explicitly:

```bash
python handler.py --test_input '{"input": {"prompt": "make it inferno", "image_base64": "<base64>"}}'
```

Or serve the local RunPod test API:

```bash
python handler.py --rp_serve_api --rp_api_port 8080 --rp_log_level DEBUG
```

## Docker Build

Build for RunPod with `linux/amd64`:

```bash
docker build --platform linux/amd64 -t yourdockerhub/flux-schnell-runpod-worker:v0.1.0 .
docker push yourdockerhub/flux-schnell-runpod-worker:v0.1.0
```

Do not deploy `latest`; use version tags.

## RunPod Deploy Notes

- Endpoint type: `Queue`
- Keep `FlashBoot` enabled
- If cold starts matter, set `Active workers` to `1+`
- Mount a network volume if you want model reuse across worker starts
- For GitHub-based RunPod deploys, create a new GitHub release when you want the endpoint updated
