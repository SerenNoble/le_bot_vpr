#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多用户集合ChromaDB存储测试
包含自动生成500ms测试音频文件的功能
"""
import os
import sys
import time
import uuid
import numpy as np
import logging
from typing import List, Dict, Any
import tempfile
import wave
from datetime import datetime

# 添加项目路径
sys.path.append('.')

from app.db.multi_collection_chroma_storage import get_multi_chroma_storage
from app.vpr.chroma_predictor import chroma_predictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioGenerator:
    """生成测试音频文件的工具类"""

    @staticmethod
    def generate_silence_wav(duration_ms: int = 500, sample_rate: int = 16000,
                          filename: str = None) -> str:
        """
        生成静音音频文件

        Args:
            duration_ms: 音频时长（毫秒）
            sample_rate: 采样率
            filename: 保存的文件名，如果为None则自动生成

        Returns:
            生成的音频文件路径
        """
        if filename is None:
            filename = f"silence_{duration_ms}ms_{int(time.time())}.wav"

        duration_samples = int(sample_rate * duration_ms / 1000)

        # 创建静音数据（全0）
        audio_data = np.zeros(duration_samples, dtype=np.int16)

        # 保存为WAV文件
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return filename

    @staticmethod
    def generate_sine_wav(frequency: float = 440, duration_ms: int = 500,
                         sample_rate: int = 16000, amplitude: float = 0.3,
                         filename: str = None) -> str:
        """
        生成正弦波音频文件

        Args:
            frequency: 频率（Hz）
            duration_ms: 音频时长（毫秒）
            sample_rate: 采样率
            amplitude: 振幅（0-1）
            filename: 保存的文件名，如果为None则自动生成

        Returns:
            生成的音频文件路径
        """
        if filename is None:
            filename = f"sine_{frequency}Hz_{duration_ms}ms_{int(time.time())}.wav"

        duration_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, duration_samples, False)

        # 生成正弦波
        audio_data = amplitude * np.sin(2 * np.pi * frequency * t)

        # 转换为16位整数
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # 保存为WAV文件
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data_int16.tobytes())

        return filename

    @staticmethod
    def generate_different_voices(num_files: int = 5, duration_ms: int = 500) -> List[str]:
        """
        生成多个不同频率的测试音频文件（模拟不同人的声音）

        Args:
            num_files: 生成的文件数量
            duration_ms: 每个文件的时长（毫秒）

        Returns:
            生成的音频文件路径列表
        """
        files = []
        frequencies = [200, 300, 400, 500, 600]  # 不同频率模拟不同人声

        for i in range(num_files):
            freq = frequencies[i % len(frequencies)]
            filename = f"voice_{i+1}_{freq}Hz_{duration_ms}ms.wav"
            filepath = AudioGenerator.generate_sine_wav(
                frequency=freq,
                duration_ms=duration_ms,
                filename=filename
            )
            files.append(filepath)
            logger.info(f"生成测试音频文件: {filepath} (频率: {freq}Hz)")

        return files


class ChromaDBTester:
    """ChromaDB多用户集合测试类"""

    def __init__(self):
        self.storage = get_multi_chroma_storage()
        self.test_files = []
        self.test_users = []

    def setup(self):
        """设置测试环境"""
        logger.info("🔧 设置测试环境...")

        # 生成测试音频文件
        logger.info("🎵 生成测试音频文件...")
        self.test_files = AudioGenerator.generate_different_voices(8, 500)

        # 创建测试用户数据
        self.test_users = [
            {
                "user_id": "test_user_zhang",
                "person_name": "张三",
                "relationship": "本人"
            },
            {
                "user_id": "test_user_zhang",
                "person_name": "张爸爸",
                "relationship": "爸爸"
            },
            {
                "user_id": "test_user_zhang",
                "person_name": "张妈妈",
                "relationship": "妈妈"
            },
            {
                "user_id": "test_user_zhang",
                "person_name": "李四",
                "relationship": "朋友"
            },
            {
                "user_id": "test_user_li",
                "person_name": "李明",
                "relationship": "本人"
            },
            {
                "user_id": "test_user_li",
                "person_name": "李妻子",
                "relationship": "妻子"
            },
            {
                "user_id": "test_user_wang",
                "person_name": "王五",
                "relationship": "本人"
            },
            {
                "user_id": "test_user_wang",
                "person_name": "王同事",
                "relationship": "同事"
            }
        ]

        logger.info(f"✅ 测试环境设置完成，生成 {len(self.test_files)} 个音频文件，{len(self.test_users)} 个测试数据")

    def cleanup(self):
        """清理测试环境"""
        logger.info("🧹 清理测试环境...")

        # 删除测试音频文件
        for file_path in self.test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"删除测试文件: {file_path}")
            except Exception as e:
                logger.warning(f"删除文件失败 {file_path}: {e}")

        self.test_files.clear()
        self.test_users.clear()
        logger.info("✅ 测试环境清理完成")

    def test_add_voice_features(self) -> bool:
        """测试添加语音特征"""
        logger.info("📝 测试添加语音特征...")

        try:
            voice_ids = []

            for i, user_data in enumerate(self.test_users):
                # 读取音频文件
                if i < len(self.test_files):
                    with open(self.test_files[i], 'rb') as f:
                        audio_data = f.read()

                    # 生成模拟特征向量 (192维，与实际模型一致)
                    feature_vector = np.random.rand(192)

                    # 添加语音特征
                    voice_id = self.storage.add_voice_feature(
                        user_id=user_data["user_id"],
                        person_name=user_data["person_name"],
                        feature_vector=feature_vector,
                        relationship=user_data["relationship"]
                    )

                    voice_ids.append(voice_id)
                    logger.info(f"添加语音特征: {user_data['user_id']} - {user_data['person_name']} ({user_data['relationship']})")

            logger.info(f"✅ 成功添加 {len(voice_ids)} 个语音特征")
            return True

        except Exception as e:
            logger.error(f"❌ 添加语音特征失败: {e}")
            return False

    def test_get_user_features(self) -> bool:
        """测试获取用户特征"""
        logger.info("🔍 测试获取用户特征...")

        try:
            # 测试张三的数据
            user_id = "test_user_zhang"
            features = self.storage.get_user_all_features(user_id)

            logger.info(f"用户 {user_id} 的特征分组: {list(features.keys())}")

            expected_persons = ["user", "张爸爸", "张妈妈", "李四"]  # 可能会有不同的UUID前缀
            found_persons = 0

            for person_key in features.keys():
                if "user" in person_key or "张爸爸" in person_key or "张妈妈" in person_key or "李四" in person_key:
                    found_persons += 1
                    logger.info(f"  - {person_key}: {len(features[person_key])} 个特征向量")

            if found_persons >= 3:  # 至少找到3个人
                logger.info("✅ 获取用户特征测试通过")
                return True
            else:
                logger.warning(f"⚠️ 获取用户特征测试部分通过，找到 {found_persons} 个人")
                return False

        except Exception as e:
            logger.error(f"❌ 获取用户特征失败: {e}")
            return False

    def test_user_persons(self) -> bool:
        """测试获取用户人员信息"""
        logger.info("👥 测试获取用户人员信息...")

        try:
            # 测试张三的人员信息
            user_id = "test_user_zhang"
            persons = self.storage.get_user_persons(user_id)

            logger.info(f"用户 {user_id} 的人员信息:")
            for person in persons:
                logger.info(f"  - 姓名: {person['person_name']}, 关系: {person['relationship']}, 音频数: {person['audio_count']}")

            # 检查是否包含爸爸、妈妈、朋友
            relationships = [person['relationship'] for person in persons]
            if any('爸爸' in r for r in relationships) and any('妈妈' in r for r in relationships):
                logger.info("✅ 获取用户人员信息测试通过")
                return True
            else:
                logger.warning("⚠️ 获取用户人员信息测试部分通过")
                return False

        except Exception as e:
            logger.error(f"❌ 获取用户人员信息失败: {e}")
            return False

    def test_user_stats(self) -> bool:
        """测试用户统计信息"""
        logger.info("📊 测试用户统计信息...")

        try:
            # 测试张三的统计信息
            user_id = "test_user_zhang"
            stats = self.storage.get_user_stats(user_id)

            logger.info(f"用户 {user_id} 统计信息:")
            logger.info(f"  - 用户音频数量: {stats['user_audio_count']}")
            logger.info(f"  - 总人员数: {stats['total_persons']}")
            logger.info(f"  - 总音频特征数: {stats['total_audio_features']}")

            if stats['user_audio_count'] >= 1 and stats['total_persons'] >= 2:
                logger.info("✅ 用户统计信息测试通过")
                return True
            else:
                logger.warning("⚠️ 用户统计信息测试部分通过")
                return False

        except Exception as e:
            logger.error(f"❌ 用户统计信息失败: {e}")
            return False

    def test_similarity_search(self) -> bool:
        """测试相似度搜索"""
        logger.info("🎯 测试相似度搜索...")

        try:
            # 创建查询向量
            query_vector = np.random.rand(192)

            # 在指定用户中搜索
            user_id = "test_user_zhang"
            results = self.storage.find_most_similar(
                query_vector=query_vector,
                user_id=user_id,
                threshold=0.1,  # 低阈值确保能找到结果
                top_k=5
            )

            logger.info(f"在用户 {user_id} 中搜索到 {len(results)} 个结果:")
            for result in results:
                logger.info(f"  - {result['person_name']} ({result['relationship']}): 相似度 {result['similarity']:.3f}")

            if len(results) >= 2:
                logger.info("✅ 相似度搜索测试通过")
                return True
            else:
                logger.warning("⚠️ 相似度搜索测试部分通过")
                return False

        except Exception as e:
            logger.error(f"❌ 相似度搜索失败: {e}")
            return False

    def test_chroma_predictor(self) -> bool:
        """测试Chroma预测器"""
        logger.info("🤖 测试Chroma预测器...")

        try:
            import asyncio

            async def test_predictor():
                # 初始化预测器
                await chroma_predictor.initialize()
                logger.info("✅ Chroma预测器初始化成功")

                # 读取测试音频文件
                if self.test_files:
                    with open(self.test_files[0], 'rb') as f:
                        audio_data = f.read()

                    # 测试注册语音
                    success, message = await chroma_predictor.register_user_voice(
                        user_id="predictor_test_user",
                        person_name="测试用户",
                        audio_data=audio_data,
                        relationship="本人"
                    )

                    if success:
                        logger.info(f"✅ 预测器注册语音成功: {message}")

                        # 测试语音识别
                        result = await chroma_predictor.recognize_user_voice(
                            audio_data=audio_data,
                            target_user_id="predictor_test_user",
                            threshold=0.1
                        )

                        logger.info(f"语音识别结果: {result.success}")
                        return True
                    else:
                        logger.error(f"❌ 预测器注册语音失败: {message}")
                        return False

                return False

            return asyncio.run(test_predictor())

        except Exception as e:
            logger.error(f"❌ Chroma预测器测试失败: {e}")
            return False

    def test_storage_info(self) -> bool:
        """测试存储信息"""
        logger.info("💾 测试存储信息...")

        try:
            info = self.storage.get_storage_info()

            logger.info("存储信息:")
            logger.info(f"  - 存储类型: {info['storage_type']}")
            logger.info(f"  - 用户数量: {info['total_users']}")
            logger.info(f"  - 基础目录: {info['base_directory']}")
            logger.info(f"  - 每用户集合数: {info['collections_per_user']}")

            if info['storage_type'] == 'single_collection_per_user':
                logger.info("✅ 存储信息测试通过")
                return True
            else:
                logger.error(f"❌ 存储类型错误: {info['storage_type']}")
                return False

        except Exception as e:
            logger.error(f"❌ 存储信息测试失败: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("🚀 开始运行所有测试...")

        results = {}

        try:
            # 设置测试环境
            self.setup()

            # 运行各项测试
            test_methods = [
                ("添加语音特征", self.test_add_voice_features),
                ("获取用户特征", self.test_get_user_features),
                ("获取用户人员", self.test_user_persons),
                ("用户统计信息", self.test_user_stats),
                ("相似度搜索", self.test_similarity_search),
                ("Chroma预测器", self.test_chroma_predictor),
                ("存储信息", self.test_storage_info),
            ]

            for test_name, test_method in test_methods:
                logger.info(f"\n{'='*50}")
                logger.info(f"🧪 运行测试: {test_name}")
                logger.info(f"{'='*50}")

                try:
                    result = test_method()
                    results[test_name] = result
                    status = "✅ 通过" if result else "❌ 失败"
                    logger.info(f"测试结果: {test_name} - {status}")
                except Exception as e:
                    results[test_name] = False
                    logger.error(f"测试异常: {test_name} - {e}")

            # 测试结果汇总
            logger.info(f"\n{'='*50}")
            logger.info("📋 测试结果汇总")
            logger.info(f"{'='*50}")

            passed = sum(1 for result in results.values() if result)
            total = len(results)

            for test_name, result in results.items():
                status = "✅ 通过" if result else "❌ 失败"
                logger.info(f"  {test_name}: {status}")

            logger.info(f"\n总体结果: {passed}/{total} 项测试通过")

            if passed == total:
                logger.info("🎉 所有测试都通过了！")
            elif passed >= total * 0.8:
                logger.info("👍 大部分测试通过了")
            else:
                logger.warning("⚠️ 有较多测试失败，需要检查")

        except Exception as e:
            logger.error(f"❌ 测试运行异常: {e}")
        finally:
            # 清理测试环境
            self.cleanup()

        return results


def main():
    """主函数"""
    print("="*60)
    print("🎵 多用户集合ChromaDB存储测试")
    print("自动生成500ms测试音频文件")
    print("="*60)

    tester = ChromaDBTester()
    results = tester.run_all_tests()

    print("\n" + "="*60)
    print("🏁 测试完成")
    print("="*60)

    # 返回退出码
    passed = sum(1 for result in results.values() if result)
    total = len(results)

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print(f"⚠️ {total-passed} 项测试失败")
        return 1


if __name__ == "__main__":
    exit(main())