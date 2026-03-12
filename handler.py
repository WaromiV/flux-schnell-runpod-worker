import os
import traceback

import runpod

from app.logic import generate_image, warmup_model


if os.getenv("PRELOAD_MODEL", "1") == "1":
    warmup_model()


def handler(job):
    job_input = job.get("input") or {}

    try:
        return generate_image(job_input)
    except Exception as exc:
        response = {
            "ok": False,
            "error": str(exc),
        }
        if os.getenv("DEBUG_ERRORS", "0") == "1":
            response["traceback"] = traceback.format_exc()
        return response


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
