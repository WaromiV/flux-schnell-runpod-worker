import base64
import io
import os
import threading
from pathlib import Path

import requests
import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-schnell")
OFFLOAD_MODE = os.getenv("OFFLOAD_MODE", "model")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "4"))
DEFAULT_STRENGTH = float(os.getenv("DEFAULT_STRENGTH", "0.95"))
DEFAULT_GUIDANCE = float(os.getenv("DEFAULT_GUIDANCE", "0.0"))
RUNPOD_VOLUME = Path(os.getenv("RUNPOD_VOLUME", "/runpod-volume"))

if RUNPOD_VOLUME.exists():
    CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", RUNPOD_VOLUME / "huggingface"))
else:
    CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/huggingface"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this worker.")

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
PIPELINE = None
PIPELINE_LOCK = threading.Lock()


def _configure_pipeline(pipe: FluxImg2ImgPipeline) -> FluxImg2ImgPipeline:
    if OFFLOAD_MODE == "none":
        pipe.to("cuda")
    elif OFFLOAD_MODE == "model":
        pipe.enable_model_cpu_offload()
    elif OFFLOAD_MODE == "sequential":
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        raise RuntimeError(f"Invalid OFFLOAD_MODE={OFFLOAD_MODE}")
    return pipe


def warmup_model() -> FluxImg2ImgPipeline:
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    with PIPELINE_LOCK:
        if PIPELINE is not None:
            return PIPELINE
        pipe = FluxImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=str(CACHE_DIR),
        )
        PIPELINE = _configure_pipeline(pipe)
        return PIPELINE


def _decode_base64_image(value: str) -> bytes:
    if not value:
        raise ValueError("image_base64 cannot be empty")
    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _load_image(job_input: dict) -> Image.Image:
    if job_input.get("image_base64"):
        raw = _decode_base64_image(job_input["image_base64"])
    elif job_input.get("image_url"):
        response = requests.get(job_input["image_url"], timeout=60)
        response.raise_for_status()
        raw = response.content
    elif job_input.get("image_path"):
        raw = Path(job_input["image_path"]).read_bytes()
    else:
        raise ValueError("Provide one of: image_base64, image_url, image_path")

    return Image.open(io.BytesIO(raw)).convert("RGB")


def _normalize_dimensions(
    image: Image.Image, width: int | None, height: int | None
) -> tuple[int, int]:
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")

    if width is not None and height is not None:
        return width, height

    src_width, src_height = image.size
    normalized_width = max(64, round(src_width / 16) * 16)
    normalized_height = max(64, round(src_height / 16) * 16)
    return normalized_width, normalized_height


def _encode_output(image: Image.Image, output_format: str) -> tuple[str, str]:
    fmt = output_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("output_format must be png, jpeg, jpg, or webp")

    buffer = io.BytesIO()
    save_kwargs = {"quality": 95} if fmt == "JPEG" else {}
    image.save(buffer, format=fmt, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "WEBP": "image/webp",
    }[fmt]
    return encoded, mime_type


def generate_image(job_input: dict) -> dict:
    prompt = (job_input.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Missing required field: prompt")

    strength = float(job_input.get("strength", DEFAULT_STRENGTH))
    steps = int(job_input.get("steps", DEFAULT_STEPS))
    guidance_scale = float(job_input.get("guidance_scale", DEFAULT_GUIDANCE))
    seed = int(job_input.get("seed", 0))
    output_format = str(job_input.get("output_format", "png"))
    width = job_input.get("width")
    height = job_input.get("height")

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if steps < 1 or steps > 8:
        raise ValueError("steps must be between 1 and 8 for FLUX.1-schnell")

    input_image = _load_image(job_input)
    width, height = _normalize_dimensions(
        input_image,
        int(width) if width is not None else None,
        int(height) if height is not None else None,
    )
    resized_image = input_image.resize((width, height))
    generator = torch.Generator("cpu").manual_seed(seed)
    pipeline = warmup_model()

    with torch.inference_mode():
        output_image = pipeline(
            prompt=prompt,
            image=resized_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            generator=generator,
        ).images[0]

    image_base64, mime_type = _encode_output(output_image, output_format)

    return {
        "ok": True,
        "model_id": MODEL_ID,
        "offload_mode": OFFLOAD_MODE,
        "seed": seed,
        "steps": steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "width": output_image.width,
        "height": output_image.height,
        "mime_type": mime_type,
        "image_base64": image_base64,
    }
