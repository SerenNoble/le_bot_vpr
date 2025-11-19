"""
多用户集合ChromaDB存储 - 每用户独立集合方案
提供更好的数据隔离和查询性能
"""
import os
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class MultiCollectionChromaStorage:
    """基于多用户集合的ChromaDB存储类"""

    def __init__(self, base_directory: str = None):
        """
        初始化多用户集合存储

        Args:
            base_directory: 基础存储目录
        """
        # 如果没有指定目录，使用临时目录
        if base_directory is None:
            import tempfile
            base_directory = tempfile.mkdtemp(prefix="voice_chroma_")
        elif not os.path.isabs(base_directory):
            # 如果是相对路径，转换为绝对路径
            base_directory = os.path.abspath(base_directory)

        self.base_directory = base_directory
        os.makedirs(base_directory, exist_ok=True)

        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=base_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=False)
        )

        # 集合缓存: {user_id: {collection_type: collection}}
        self.collections_cache = {}

        logger.info(f"✅ 多用户集合ChromaDB初始化完成: {base_directory}")

    def _get_user_collection(self, user_id: str):
        """
        获取用户的专属集合（每用户一个集合）

        Args:
            user_id: 用户ID

        Returns:
            ChromaDB集合对象
        """
        if user_id not in self.collections_cache:
            collection_name = f"user_{user_id}_voice_features"

            # 创建或获取集合
            self.collections_cache[user_id] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "user_id": user_id}
            )

            logger.info(f"✅ 创建/获取用户集合: {collection_name}")

        return self.collections_cache[user_id]

    def add_voice_feature(self, user_id: str, person_name: str, feature_vector: np.ndarray, relationship: str = "朋友") -> str:
        """
        添加语音特征向量到用户专属集合

        Args:
            user_id: 用户唯一标识
            person_name: 人员姓名（实际存储的姓名，如"张三"、"李四"等）
            feature_vector: 特征向量
            relationship: 与用户的关系（如"本人"、"爸爸"、"妈妈"、"朋友"、"同事"等）

        Returns:
            voice_id: 生成的语音特征唯一ID
        """
        try:
            # 生成唯一ID
            voice_id = f"{user_id}_{person_name}_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # 根据关系字段判断是否是用户本人
            is_user = (relationship.lower() in ["本人", "self", "me", "本人"])

            # 获取用户专属集合
            collection = self._get_user_collection(user_id)

            # 添加到集合（所有特征都在一个集合中）
            collection.add(
                embeddings=[feature_vector.tolist()],
                metadatas=[{
                    "user_id": user_id,           # 用户ID
                    "person_name": person_name,     # 实际存储的人员姓名
                    "relationship": relationship,   # 与用户的关系
                    "is_user": str(is_user),       # 是否是用户本人（根据关系判断）
                    "created_at": datetime.now().isoformat(),
                    "created_timestamp": int(time.time() * 1000)
                }],
                ids=[voice_id]
            )

            logger.info(f"✅ 已添加语音特征: {voice_id} (用户: {user_id}, 人员: {person_name}, 关系: {relationship}, 类型: {'本人' if is_user else '人员'})")
            return voice_id

        except Exception as e:
            logger.error(f"❌ 添加语音特征失败: {e}")
            raise

    def get_user_all_features(self, user_id: str) -> Dict[str, List[np.ndarray]]:
        """
        获取指定用户的所有语音特征向量

        Args:
            user_id: 用户ID

        Returns:
            按人员分组的特征向量字典
            格式: {"user": [特征向量], "妈妈": [特征向量1, 特征向量2], ...}
        """
        try:
            # 获取用户专属集合
            collection = self._get_user_collection(user_id)
            results = collection.get(include=["embeddings", "metadatas"])

            # 按人员分组
            features_by_person = {}

            for i, voice_id in enumerate(results['ids']):
                if i >= len(results['metadatas']):
                    continue

                metadata = results['metadatas'][i]
                embedding = results['embeddings'][i]

                person_name = metadata['person_name']
                is_user = metadata['is_user'] == 'True'

                # 转换为numpy数组
                feature_vector = np.array(embedding)

                # 确定人员键名
                if is_user:
                    person_key = "user"
                else:
                    person_key = f"{voice_id.split('_')[2]}_{person_name}"

                # 按人员分组
                if person_key not in features_by_person:
                    features_by_person[person_key] = []
                features_by_person[person_key].append(feature_vector)

            logger.info(f"📊 获取用户 {user_id} 的特征: {len(features_by_person)} 个人员")
            return features_by_person

        except Exception as e:
            logger.error(f"❌ 获取用户特征失败: {e}")
            return {}

    def find_most_similar(self, query_vector: np.ndarray,
                         user_id: str = None,
                         threshold: float = 0.6,
                         top_k: int = 10) -> List[Dict[str, Any]]:
        """
        查找最相似的语音特征

        Args:
            query_vector: 查询向量
            user_id: 指定用户ID（如果为None则在所有用户中搜索）
            threshold: 相似度阈值
            top_k: 返回结果数量

        Returns:
            相似结果列表
        """
        try:
            similar_voices = []

            if user_id:
                # 在指定用户的单个集合中搜索
                collection = self._get_user_collection(user_id)
                results = collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    include=["metadatas", "distances"]
                )

                # 格式化结果
                for i in range(len(results['ids'][0])):
                    voice_id = results['ids'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]  # 余弦距离 (0-2)
                    similarity = 1 - distance/2  # 转换为余弦相似度 (0-1)

                    # 过滤低相似度结果
                    if similarity >= threshold:
                        similar_voices.append({
                            "voice_id": voice_id,
                            "user_id": metadata['user_id'],
                            "person_id": voice_id,
                            "person_name": metadata['person_name'],
                            "relationship": metadata.get('relationship', '朋友'),  # 添加关系字段
                            "is_user": metadata['is_user'] == 'True',
                            "similarity": similarity,
                            "distance": distance,
                            "created_at": metadata['created_at']
                        })
            else:
                # 在所有用户的集合中搜索
                if os.path.exists(self.base_directory):
                    user_dirs = [d for d in os.listdir(self.base_directory)
                               if d.startswith("user_") and os.path.isdir(os.path.join(self.base_directory, d))]

                    for user_dir in user_dirs:
                        target_user_id = user_dir[5:]  # 去掉"user_"前缀
                        try:
                            collection = self._get_user_collection(target_user_id)
                            results = collection.query(
                                query_embeddings=[query_vector.tolist()],
                                n_results=top_k,
                                include=["metadatas", "distances"]
                            )

                            for i in range(len(results['ids'][0])):
                                voice_id = results['ids'][0][i]
                                metadata = results['metadatas'][0][i]
                                distance = results['distances'][0][i]
                                similarity = 1 - distance/2

                                if similarity >= threshold:
                                    similar_voices.append({
                                        "voice_id": voice_id,
                                        "user_id": metadata['user_id'],
                                        "person_id": voice_id,
                                        "person_name": metadata['person_name'],
                                        "relationship": metadata.get('relationship', '朋友'),  # 添加关系字段
                                        "is_user": metadata['is_user'] == 'True',
                                        "similarity": similarity,
                                        "distance": distance,
                                        "created_at": metadata['created_at']
                                    })
                        except:
                            continue

            # 按相似度排序并限制数量
            similar_voices.sort(key=lambda x: x["similarity"], reverse=True)
            similar_voices = similar_voices[:top_k]

            logger.info(f"🔍 相似度搜索完成: 找到 {len(similar_voices)} 个匹配结果 (阈值: {threshold})")
            return similar_voices

        except Exception as e:
            logger.error(f"❌ 相似度搜索失败: {e}")
            return []

    
    def delete_user_all_voices(self, user_id: str) -> int:
        """
        删除用户的所有语音数据

        Args:
            user_id: 用户ID

        Returns:
            删除的语音数量
        """
        try:
            # 获取用户的集合
            try:
                collection = self._get_user_collection(user_id)
                # 获取集合中所有的语音ID
                results = collection.get()
                voice_ids = results['ids']

                if voice_ids:
                    # 删除所有语音特征
                    collection.delete(ids=voice_ids)
                    deleted_count = len(voice_ids)
                    logger.info(f"✅ 已删除用户 {user_id} 的 {deleted_count} 个语音特征")
                    return deleted_count
                else:
                    logger.info(f"⚠️ 用户 {user_id} 没有语音数据")
                    return 0

            except Exception as e:
                logger.error(f"❌ 获取用户 {user_id} 的集合失败: {e}")
                return 0

        except Exception as e:
            logger.error(f"❌ 删除用户语音失败: {e}")
            return 0

    def get_user_stats(self, user_id: str = None) -> Dict[str, Any]:
        """获取用户统计信息"""
        try:
            if user_id:
                # 获取特定用户统计
                try:
                    collection = self._get_user_collection(user_id)
                    results = collection.get(include=["metadatas"])

                    user_audio_count = 0
                    persons_info = {}

                    for i, voice_id in enumerate(results['ids']):
                        if i >= len(results['metadatas']):
                            continue

                        metadata = results['metadatas'][i]
                        person_name = metadata['person_name']
                        is_user = metadata['is_user'] == 'True'

                        if is_user:
                            user_audio_count += 1
                        else:
                            if person_name not in persons_info:
                                persons_info[person_name] = {
                                    "person_id": f"{user_id}_{person_name}_{int(time.time())}",
                                    "person_name": person_name,
                                    "audio_count": 0,
                                    "created_at": metadata['created_at']
                                }
                            persons_info[person_name]["audio_count"] += 1

                    return {
                        "user_id": user_id,
                        "user_audio_count": user_audio_count,
                        "total_persons": len(persons_info),
                        "total_audio_features": user_audio_count + sum(p["audio_count"] for p in persons_info.values()),
                        "persons_detail": list(persons_info.values()),
                        "last_updated": datetime.utcnow()
                    }
                except:
                    return {
                        "user_id": user_id,
                        "user_audio_count": 0,
                        "total_persons": 0,
                        "total_audio_features": 0,
                        "persons_detail": [],
                        "last_updated": datetime.utcnow()
                    }
            else:
                # 获取全局统计
                all_stats = {
                    "total_users": 0,
                    "total_persons": 0,
                    "total_audio_features": 0
                }

                # 扫描所有用户目录
                if os.path.exists(self.base_directory):
                    for user_dir in os.listdir(self.base_directory):
                        if user_dir.startswith("user_") and os.path.isdir(os.path.join(self.base_directory, user_dir)):
                            target_user_id = user_dir[5:]  # 去掉"user_"前缀
                            all_stats["total_users"] += 1

                            # 统计该用户的数据
                            try:
                                collection = self._get_user_collection(target_user_id)
                                count = collection.count()
                                all_stats["total_audio_features"] += count

                                # 简化统计：如果用户有数据，假设至少有一个人员
                                if count > 0:
                                    results = collection.get(include=["metadatas"])
                                    has_user_voice = any(m['is_user'] == 'True' for m in results['metadatas'])
                                    has_persons_voice = any(m['is_user'] == 'False' for m in results['metadatas'])

                                    if has_user_voice and has_persons_voice:
                                        all_stats["total_persons"] += 1
                                    elif has_persons_voice:
                                        # 统计不同人员数量
                                        person_names = set(m['person_name'] for m in results['metadatas'] if m['is_user'] == 'False')
                                        all_stats["total_persons"] += len(person_names)
                            except:
                                pass

                all_stats["last_updated"] = datetime.utcnow()
                return all_stats

        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            return {}

    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息"""
        try:
            user_count = 0
            if os.path.exists(self.base_directory):
                user_dirs = [d for d in os.listdir(self.base_directory)
                           if d.startswith("user_") and os.path.isdir(os.path.join(self.base_directory, d))]
                user_count = len(user_dirs)

            return {
                "storage_type": "single_collection_per_user",
                "total_users": user_count,
                "base_directory": self.base_directory,
                "hnsw_space": "cosine",
                "collections_per_user": 1  # 每用户一个集合
            }
        except Exception as e:
            logger.error(f"❌ 获取存储信息失败: {e}")
            return {}

    def search_users(self, limit: int = 100) -> List[Dict[str, Any]]:
        """搜索用户"""
        try:
            users_list = []

            if os.path.exists(self.base_directory):
                user_dirs = [d for d in os.listdir(self.base_directory)
                           if d.startswith("user_") and os.path.isdir(os.path.join(self.base_directory, d))]

                for user_dir in user_dirs[:limit]:
                    user_id = user_dir[5:]  # 去掉"user_"前缀

                    # 统计用户数据
                    try:
                        collection = self._get_user_collection(user_id)
                        results = collection.get(include=["metadatas"])

                        user_audio_count = 0
                        persons_names = set()
                        created_ats = []

                        for metadata in results['metadatas']:
                            is_user = metadata['is_user'] == 'True'
                            if is_user:
                                user_audio_count += 1
                            else:
                                persons_names.add(metadata['person_name'])
                            created_ats.append(metadata['created_at'])

                        users_list.append({
                            "user_id": user_id,
                            "user_name": None,
                            "user_audio_count": user_audio_count,
                            "total_persons": len(persons_names),
                            "created_at": min(created_ats) if created_ats else "unknown"
                        })

                    except:
                        users_list.append({
                            "user_id": user_id,
                            "user_name": None,
                            "user_audio_count": 0,
                            "total_persons": 0,
                            "created_at": "unknown"
                        })

            logger.info(f"👥 搜索到 {len(users_list)} 个用户")
            return users_list

        except Exception as e:
            logger.error(f"❌ 搜索用户失败: {e}")
            return []

    def get_user_persons(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有人员信息"""
        try:
            persons = {}

            # 从用户集合中获取所有人员数据（排除本人）
            try:
                collection = self._get_user_collection(user_id)
                results = collection.get(include=["metadatas"])

                for i, voice_id in enumerate(results['ids']):
                    if i >= len(results['metadatas']):
                        continue

                    metadata = results['metadatas'][i]
                    person_name = metadata['person_name']
                    relationship = metadata.get('relationship', '朋友')
                    is_user = metadata['is_user'] == 'True'

                    # 跳过用户本人，只统计其他人员
                    if is_user:
                        continue

                    created_at = metadata['created_at']

                    if person_name not in persons:
                        persons[person_name] = {
                            "person_id": f"{user_id}_{person_name}_{int(time.time())}",
                            "person_name": person_name,
                            "relationship": relationship,  # 添加关系字段
                            "audio_count": 0,
                            "created_at": created_at
                        }

                    persons[person_name]["audio_count"] += 1

            except Exception as e:
                logger.debug(f"用户 {user_id} 暂无人员数据: {e}")

            return list(persons.values())

        except Exception as e:
            logger.error(f"❌ 获取用户人员失败: {e}")
            return []

    def delete_user_person_voices(self, user_id: str, person_name: str) -> int:
        """删除用户的特定人员语音"""
        try:
            collection = self._get_user_collection(user_id)

            # 获取该人员的所有语音（在用户集合中查找）
            results = collection.get()
            voice_ids_to_delete = []

            for i, voice_id in enumerate(results['ids']):
                if i >= len(results['metadatas']):
                    continue

                metadata = results['metadatas'][i]
                if metadata['person_name'] == person_name and metadata['is_user'] == 'False':
                    voice_ids_to_delete.append(voice_id)

            if voice_ids_to_delete:
                collection.delete(ids=voice_ids_to_delete)
                logger.info(f"✅ 已删除 {len(voice_ids_to_delete)} 个语音特征 (用户: {user_id}, 人员: {person_name})")
                return len(voice_ids_to_delete)

            logger.warning(f"⚠️ 未找到要删除的语音 (用户: {user_id}, 人员: {person_name})")
            return 0

        except Exception as e:
            logger.error(f"❌ 删除人员语音失败: {e}")
            return 0

    def clear_all(self) -> bool:
        """清空所有数据"""
        try:
            # 获取所有集合并删除
            collections = self.client.list_collections()
            for collection in collections:
                self.client.delete_collection(collection.name)

            # 清空缓存
            self.collections_cache.clear()

            logger.warning("⚠️ 已清空所有集合")
            return True
        except Exception as e:
            logger.error(f"❌ 清空数据失败: {e}")
            return False


# 全局存储实例
_multi_chroma_storage = None

def get_multi_chroma_storage() -> MultiCollectionChromaStorage:
    """获取全局多用户集合ChromaDB存储实例"""
    global _multi_chroma_storage
    if _multi_chroma_storage is None:
        _multi_chroma_storage = MultiCollectionChromaStorage()
    return _multi_chroma_storage