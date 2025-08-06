import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deps", "vpr"
    )
)

from deps.vpr.mvector.predict import MVectorPredictor

predictor = MVectorPredictor(
    configs="app/vpr/configs/eres2net.yml",
    model_path="app/vpr/models/ERes2Net_Fbank/best_model/",
    use_gpu=False,
)

# export the predictor for use in other modules
__all__ = ["predictor"]
