# Warning suppression MUST happen before all other imports,
# so that TF_CPP_MIN_LOG_LEVEL takes effect.
from services.warnings_config import configure_warnings
configure_warnings()

# --- GPU mixed precision + cuDNN autotuner (Issue #35) ---
# Must happen BEFORE the model imports (via routes -> walk_forward ->
# lstm/transformer), so that Keras/PyTorch know the global policies before the
# first layer is built. On CPU-only systems, mixed_float16 is skipped.
import tensorflow as tf
if tf.config.list_physical_devices("GPU"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

import torch
torch.backends.cudnn.benchmark = True

from fastapi import FastAPI
from services.logging_config import setup_service_logger
from services.model_service.routes import router

logger = setup_service_logger("model_service")

app = FastAPI(title="Model Service", version="0.1.0")
app.include_router(router)

@app.on_event("startup")
def startup():
    logger.info("Model Service started")
