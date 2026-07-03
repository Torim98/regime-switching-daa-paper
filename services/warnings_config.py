"""Central warning suppression for all microservices.

Suppresses expected, harmless warnings from statsmodels, Keras, and
TensorFlow that would otherwise flood the service logs. All suppressed
warnings have been reviewed and are non-critical in the context of the
regime-switching pipeline.
"""
import os
import warnings
import logging


def configure_warnings():
    # --- TensorFlow ---
    # Must be set BEFORE the first TF import
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

    # --- statsmodels (MSM, Markov switching) ---
    try:
        from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
        warnings.filterwarnings("ignore", category=ValueWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass
    warnings.filterwarnings("ignore", message="A date index has been provided")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
    warnings.filterwarnings("ignore", message="Model is not converging")
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="statsmodels")

    # --- Keras / TensorFlow ---
    warnings.filterwarnings("ignore", message="Do not pass an `input_shape`")
    warnings.filterwarnings("ignore", message="triggered tf.function retracing")