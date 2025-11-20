import sys
import os
import time
from functools import wraps
from loguru import logger

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deps", "vpr"
    )
)

from deps.vpr.mvector.predict import MVectorPredictor

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 使用绝对路径
config_path = os.path.join(project_root, "app", "vpr", "configs", "eres2net.yml")
model_path = os.path.join(project_root, "app", "vpr", "models", "ERes2Net_Fbank", "best_model")

print(f"MongoDB Voice Recognition System Initializing...")
print(f"   Config path: {config_path}")
print(f"   Model path: {model_path}")

# 检查文件是否存在
if not os.path.exists(config_path):
    print(f"Config file not found: {config_path}")
    # 尝试使用相对路径作为备选
    config_path = "eres2net"
    print(f"Trying config name: {config_path}")

if not os.path.exists(model_path):
    print(f"Model directory not found: {model_path}")
    print("   Please ensure model files are downloaded and placed")

# 初始化基础预测器（用于特征提取）
print("Loading MVectorPredictor for feature extraction...")
predictor = MVectorPredictor(
    configs=config_path,
    model_path=model_path,
    use_gpu=False,
    threshold=0.6
)
print("MVectorPredictor loaded for MongoDB system")

# 导出的预测器
__all__ = [
    "predictor"           # MVectorPredictor（用于特征提取）
]
