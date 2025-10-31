import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deps", "vpr"
    )
)

from deps.vpr.mvector.predict import MVectorPredictor

import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# 创建音频数据库目录
audio_db_path = os.path.join(project_root, "audio_db")
os.makedirs(audio_db_path, exist_ok=True)

# 使用绝对路径
config_path = os.path.join(project_root, "app", "vpr", "configs", "eres2net.yml")
model_path = os.path.join(project_root, "app", "vpr", "models", "ERes2Net_Fbank", "best_model")

print(f"配置文件路径: {config_path}")
print(f"模型路径: {model_path}")
print(f"音频数据库路径: {audio_db_path}")

# 检查文件是否存在
if not os.path.exists(config_path):
    print(f"❌ 配置文件不存在: {config_path}")
    # 尝试使用相对路径作为备选
    config_path = "eres2net"
    print(f"🔄 尝试使用配置名称: {config_path}")

if not os.path.exists(model_path):
    print(f"⚠️  模型目录不存在: {model_path}")
    print("   请确保已下载并放置模型文件")

predictor = MVectorPredictor(
    configs=config_path,
    model_path=model_path,
    use_gpu=False,
    audio_db_path=audio_db_path,
    threshold=0.6
)

# export the predictor for use in other modules
__all__ = ["predictor"]
