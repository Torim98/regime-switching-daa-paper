# Warning-Suppression MUSS vor allen anderen Imports passieren,
# damit TF_CPP_MIN_LOG_LEVEL greift.
from services.warnings_config import configure_warnings
configure_warnings()

# --- GPU Mixed-Precision + cuDNN-Autotuner (Issue #35) ---
# Muss VOR den Modell-Imports (via routes -> walk_forward -> lstm/transformer)
# passieren, damit Keras/PyTorch die globalen Policies kennen, bevor der erste
# Layer gebaut wird. Auf CPU-only Systemen wird mixed_float16 uebersprungen.
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
