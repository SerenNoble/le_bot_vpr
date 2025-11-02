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


class TimedMVectorPredictor:
    """MVectorPredictor的高精度计时包装器"""

    def __init__(self, predictor_instance):
        self._predictor = predictor_instance
        self.timing_stats = {}

    def _time_method(self, method_name, method, *args, **kwargs):
        """使用高精度计时器记录方法执行时间"""
        start_time = time.perf_counter()
        try:
            result = method(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 记录统计信息
            if method_name not in self.timing_stats:
                self.timing_stats[method_name] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0,
                    'last_time': 0.0
                }

            stats = self.timing_stats[method_name]
            stats['count'] += 1
            stats['total_time'] += elapsed_time
            stats['min_time'] = min(stats['min_time'], elapsed_time)
            stats['max_time'] = max(stats['max_time'], elapsed_time)
            stats['last_time'] = elapsed_time

            # 记录日志
            logger.info(f"⏱️  {method_name} 执行时间: {elapsed_time*1000:.2f} ms ({elapsed_time:.6f} s)")

            return result
        except Exception as e:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            logger.error(f"❌ {method_name} 执行失败 (耗时: {elapsed_time*1000:.2f} ms): {str(e)}")
            raise

    def register(self, audio_data, user_name: str, sample_rate=16000):
        """声纹注册（带计时）"""
        return self._time_method('register', self._predictor.register,
                                audio_data, user_name, sample_rate)

    def recognition(self, audio_data, threshold=None, sample_rate=16000):
        """声纹识别（带计时）"""
        return self._time_method('recognition', self._predictor.recognition,
                                audio_data, threshold, sample_rate)

    def contrast(self, audio_data1, audio_data2):
        """声纹对比（带计时）"""
        return self._time_method('contrast', self._predictor.contrast,
                                audio_data1, audio_data2)

    def predict(self, audio_data, sample_rate=16000):
        """预测特征向量（带计时）"""
        return self._time_method('predict', self._predictor.predict,
                                audio_data, sample_rate)

    def predict_batch(self, audios_data, sample_rate=16000, batch_size=32):
        """批量预测特征向量（带计时）"""
        return self._time_method('predict_batch', self._predictor.predict_batch,
                                audios_data, sample_rate, batch_size)

    def get_users(self):
        """获取所有用户（带计时）"""
        return self._time_method('get_users', self._predictor.get_users)

    def remove_user(self, user_name):
        """删除用户（带计时）"""
        return self._time_method('remove_user', self._predictor.remove_user, user_name)

    def get_timing_stats(self):
        """获取计时统计信息"""
        return self.timing_stats

    def print_timing_stats(self):
        """打印计时统计信息"""
        if not self.timing_stats:
            print("\n📊 暂无计时统计数据")
            return

        print("\n" + "="*80)
        print("📊 方法执行时间统计")
        print("="*80)
        print(f"{'方法名':<20} {'调用次数':<10} {'总时间(ms)':<15} {'平均(ms)':<15} {'最小(ms)':<15} {'最大(ms)':<15}")
        print("-"*80)

        for method_name, stats in sorted(self.timing_stats.items()):
            avg_time = (stats['total_time'] / stats['count']) * 1000
            print(f"{method_name:<20} {stats['count']:<10} "
                  f"{stats['total_time']*1000:<15.2f} {avg_time:<15.2f} "
                  f"{stats['min_time']*1000:<15.2f} {stats['max_time']*1000:<15.2f}")

        print("="*80)

    def reset_timing_stats(self):
        """重置计时统计"""
        self.timing_stats = {}
        logger.info("🔄 计时统计已重置")

    # 代理其他属性到原始predictor
    def __getattr__(self, name):
        return getattr(self._predictor, name)


# 创建原始predictor实例
_base_predictor = MVectorPredictor(
    configs=config_path,
    model_path=model_path,
    use_gpu=False,
    audio_db_path=audio_db_path,
    threshold=0.6
)

# 使用计时包装器包装predictor
predictor = TimedMVectorPredictor(_base_predictor)

logger.info("✅ 已启用高精度方法计时功能")

# export the predictor for use in other modules
__all__ = ["predictor"]
