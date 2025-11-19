"""
ChromaDB声纹识别器 - 使用ChromaDB替代MongoDB
支持多用户管理和高性能向量检索
"""
import os
import sys
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import logging

# 添加依赖路径
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "deps", "vpr"
    )
)

from app.db.multi_collection_chroma_storage import get_multi_chroma_storage, MultiCollectionChromaStorage
from app.models.chroma_models import RecognitionResult, UserStats

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVoicePredictor:
    """基于ChromaDB的声纹识别器"""

    def __init__(self, lazy_load: bool = True, cache_timeout: int = 300):
        """
        初始化ChromaDB声纹识别器

        Args:
            lazy_load: 是否启用延迟加载
            cache_timeout: 缓存超时时间（秒）
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        # 初始化基础预测器
        from app.vpr import predictor
        self.base_predictor = predictor

        # 多用户集合ChromaDB存储
        self.chroma_storage: Optional[MultiCollectionChromaStorage] = None

        # 缓存设置（保留用于临时优化）
        self.lazy_load = lazy_load
        self.cache_timeout = cache_timeout
        self.feature_cache = {}  # 格式: {user_id: {person_name: [feature_vectors]}}
        self.user_profile_cache = {}  # 用户档案缓存

        logger.info("✅ ChromaDB声纹识别器初始化完成")

    async def initialize(self):
        """异步初始化方法"""
        self.chroma_storage = get_multi_chroma_storage()
        logger.info("✅ 多用户集合ChromaDB连接初始化完成")

    def _extract_features(self, audio_data: bytes) -> np.ndarray:
        """提取音频特征"""
        try:
            return self.base_predictor.predict(audio_data=audio_data)
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise

    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return (time.time() - timestamp) < self.cache_timeout

    async def _get_cached_features(self, user_id: str) -> Dict[str, List[np.ndarray]]:
        """获取缓存的特征"""
        if not self.lazy_load:
            return {}

        cached_user = self.feature_cache.get(user_id, {})
        if not cached_user:
            return {}

        current_time = time.time()
        result = {}

        for person_name, data in cached_user.items():
            if self._is_cache_valid(data["timestamp"]):
                result[person_name] = data["features"]

        return result

    async def _cache_features(self, user_id: str, features_by_person: Dict[str, List[np.ndarray]]):
        """缓存特征"""
        if not self.lazy_load:
            return

        if user_id not in self.feature_cache:
            self.feature_cache[user_id] = {}

        current_time = time.time()
        for person_name, features in features_by_person.items():
            self.feature_cache[user_id][person_name] = {
                "features": features,
                "timestamp": current_time
            }

        total_features = sum(len(features) for features in features_by_person.values())
        logger.info(f"✅ 缓存用户 {user_id} 的 {total_features} 个特征")

    async def _clear_user_cache(self, user_id: str):
        """清空用户缓存"""
        if user_id in self.feature_cache:
            del self.feature_cache[user_id]
        if user_id in self.user_profile_cache:
            del self.user_profile_cache[user_id]

    async def register_user_voice(self, user_id: str, person_name: str,
                                 audio_data: bytes, relationship: str = "朋友") -> Tuple[bool, str]:
        """
        注册用户音频

        Args:
            user_id: 用户唯一标识
            person_name: 人员姓名（如：张三、李四等实际姓名）
            relationship: 与用户的关系（如："本人"、"爸爸"、"妈妈"、"朋友"、"同事"等）
            audio_data: 音频数据字节

        Returns:
            Tuple[success: bool, message: str]
        """
        try:
            # 确保初始化完成
            if not self.chroma_storage:
                await self.initialize()

            # 提取音频特征
            logger.info(f"为用户 {user_id} 人员 {person_name} (关系: {relationship}) 提取特征...")
            feature_vector = self._extract_features(audio_data)

            # 存储到ChromaDB
            voice_id = self.chroma_storage.add_voice_feature(
                user_id=user_id,
                person_name=person_name,
                feature_vector=feature_vector,
                relationship=relationship
            )

            # 清空相关缓存
            await self._clear_user_cache(user_id)

            # 根据关系确定人员类型
            person_type = relationship if relationship.lower() in ["本人", "self", "me"] else f"{relationship}({person_name})"

            logger.info(f"✅ 成功注册用户 {user_id} {person_type} 的音频")
            return True, f"成功注册 {person_name} ({relationship}) 的音频特征"

        except Exception as e:
            error_msg = f"注册音频失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    async def recognize_user_voice(self, audio_data: bytes, threshold: float = 0.6,
                                  target_user_id: str = None) -> RecognitionResult:
        """
        识别用户音频

        Args:
            audio_data: 音频数据字节
            threshold: 识别阈值
            target_user_id: 目标用户ID（如果指定，只在该用户下搜索）

        Returns:
            RecognitionResult: 识别结果
        """
        start_time = time.time()

        try:
            # 确保初始化完成
            if not self.chroma_storage:
                await self.initialize()

            # 提取音频特征
            logger.info("提取待识别音频特征...")
            query_vector = self._extract_features(audio_data)

            # 在ChromaDB中搜索相似向量
            logger.info(f"在 {'用户' + target_user_id if target_user_id else '所有用户'} 中搜索...")
            similar_results = self.chroma_storage.find_most_similar(
                query_vector=query_vector,
                user_id=target_user_id,
                threshold=threshold,
                top_k=10
            )

            processing_time = (time.time() - start_time) * 1000

            if similar_results:
                # 取最佳匹配
                best_match = similar_results[0]
                match_details = {
                    "similarity": best_match["similarity"],
                    "distance": best_match["distance"],
                    "voice_id": best_match["voice_id"],
                    "searched_type": "specific_user" if target_user_id else "all_users",
                    "total_candidates": len(similar_results)
                }

                person_type = "本人" if best_match["is_user"] else "人员"
                logger.info(f"识别成功: {best_match['person_name']} ({person_type}) - 置信度: {best_match['similarity']:.3f}")

                return RecognitionResult(
                    success=True,
                    user_id=best_match["user_id"],
                    voice_id=best_match["voice_id"],
                    person_id=best_match["voice_id"],
                    person_name=best_match["person_name"],
                    is_user=best_match["is_user"],
                    confidence=best_match["similarity"],
                    similarity=best_match["similarity"],
                    distance=best_match["distance"],
                    match_details=match_details,
                    processing_time_ms=processing_time
                )
            else:
                logger.info(f"未找到匹配的用户 (阈值: {threshold})")
                return RecognitionResult(
                    success=False,
                    confidence=0.0,
                    match_details={
                        "threshold": threshold,
                        "searched_type": "specific_user" if target_user_id else "all_users"
                    },
                    processing_time_ms=processing_time
                )

        except Exception as e:
            error_msg = f"音频识别失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return RecognitionResult(
                success=False,
                match_details={"error": error_msg},
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def get_user_persons(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有人员"""
        if not self.chroma_storage:
            await self.initialize()

        persons = self.chroma_storage.get_user_persons(user_id)
        return [
            {
                "person_id": person["person_id"],
                "person_name": person["person_name"],
                "audio_count": person["audio_count"],
                "created_at": person["created_at"]
            }
            for person in persons
        ]

    async def delete_user(self, user_id: str) -> Tuple[bool, str]:
        """删除用户及其所有数据"""
        try:
            if not self.chroma_storage:
                await self.initialize()

            deleted_count = self.chroma_storage.delete_user_all_voices(user_id)
            await self._clear_user_cache(user_id)

            if deleted_count > 0:
                logger.info(f"✅ 成功删除用户 {user_id} 的 {deleted_count} 个语音特征")
                return True, f"用户 {user_id} 及其 {deleted_count} 个语音特征已删除"
            else:
                return False, f"用户 {user_id} 不存在或无数据"

        except Exception as e:
            error_msg = f"删除用户失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    async def delete_person(self, user_id: str, person_id: str) -> Tuple[bool, str]:
        """删除用户下的特定人员"""
        try:
            if not self.chroma_storage:
                await self.initialize()

            # 从person_id中提取person_name
            # 格式：{user_id}_{person_name}_{timestamp}
            parts = person_id.split('_')
            if len(parts) >= 3:
                person_name = '_'.join(parts[1:-1])  # 取中间部分作为person_name
            else:
                person_name = person_id  # 如果格式不符，直接使用整个id

            deleted_count = self.chroma_storage.delete_user_person_voices(user_id, person_name)
            await self._clear_user_cache(user_id)

            if deleted_count > 0:
                logger.info(f"✅ 成功删除用户 {user_id} 人员 {person_name} 的 {deleted_count} 个语音特征")
                return True, f"人员 {person_name} 已删除"
            else:
                return False, "人员不存在或无数据"

        except Exception as e:
            error_msg = f"删除人员失败: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    async def get_user_stats(self, user_id: str = None) -> Dict[str, Any]:
        """获取用户统计信息"""
        if not self.chroma_storage:
            await self.initialize()

        if user_id:
            # 获取特定用户统计
            stats = self.chroma_storage.get_user_stats(user_id)
        else:
            # 获取全局统计
            stats = self.chroma_storage.get_user_stats()

            # 添加存储状态信息
            storage_info = self.chroma_storage.get_storage_info()
            stats.update({
                "storage_info": storage_info,
                "cache_status": {
                    "cached_users": len(self.feature_cache),
                    "cache_timeout": self.cache_timeout
                }
            })

        return stats

    async def search_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """搜索用户"""
        if not self.chroma_storage:
            await self.initialize()

        users = self.chroma_storage.search_users(limit=limit)
        return users

    async def clear_cache(self):
        """清空所有缓存"""
        self.feature_cache.clear()
        self.user_profile_cache.clear()
        logger.info("✅ 所有缓存已清空")

    async def clear_all_data(self):
        """清空所有数据（危险操作）"""
        if not self.chroma_storage:
            await self.initialize()

        success = self.chroma_storage.clear_all()
        await self.clear_cache()

        if success:
            logger.warning("⚠️ 所有用户集合数据已清空")
            return True
        return False

    async def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        if not self.chroma_storage:
            await self.initialize()

        return self.chroma_storage.get_storage_info()


# 全局ChromaDB预测器实例
chroma_predictor = ChromaVoicePredictor()