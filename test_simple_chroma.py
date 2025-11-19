#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版多用户集合ChromaDB存储测试
包含自动生成500ms测试音频文件的功能
"""
import os
import sys
import time
import numpy as np
import logging
from typing import List
import wave

# 添加项目路径
sys.path.append('.')

from app.db.multi_collection_chroma_storage import get_multi_chroma_storage

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_test_wav(filename: str, duration_ms: int = 500, frequency: float = 440) -> str:
    """生成测试音频文件"""
    sample_rate = 16000
    duration_samples = int(sample_rate * duration_ms / 1000)

    # 生成正弦波
    t = np.linspace(0, duration_ms / 1000, duration_samples, False)
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)

    # 转换为16位整数
    audio_data_int16 = (audio_data * 32767).astype(np.int16)

    # 保存为WAV文件
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data_int16.tobytes())

    return filename


def test_chroma_storage():
    """测试ChromaDB存储功能"""
    print("="*60)
    print("多用户集合ChromaDB存储测试")
    print("自动生成500ms测试音频文件")
    print("="*60)

    try:
        # 初始化存储
        storage = get_multi_chroma_storage()
        print("✓ ChromaDB存储初始化成功")

        # 生成测试音频文件
        print("\n生成测试音频文件...")
        test_files = []
        frequencies = [200, 300, 400, 500]

        for i, freq in enumerate(frequencies):
            filename = f"test_audio_{i+1}_{freq}Hz.wav"
            filepath = generate_test_wav(filename, 500, freq)
            test_files.append(filepath)
            print(f"✓ 生成音频文件: {filename} (频率: {freq}Hz)")

        # 测试数据
        test_data = [
            {"user_id": "user001", "person_name": "张三", "relationship": "本人"},
            {"user_id": "user001", "person_name": "张爸爸", "relationship": "爸爸"},
            {"user_id": "user001", "person_name": "张妈妈", "relationship": "妈妈"},
            {"user_id": "user002", "person_name": "李四", "relationship": "本人"},
            {"user_id": "user002", "person_name": "李妻子", "relationship": "妻子"},
        ]

        # 测试添加语音特征
        print("\n测试添加语音特征...")
        voice_ids = []

        for i, data in enumerate(test_data):
            # 生成模拟特征向量 (192维)
            feature_vector = np.random.rand(192)

            # 添加语音特征
            voice_id = storage.add_voice_feature(
                user_id=data["user_id"],
                person_name=data["person_name"],
                feature_vector=feature_vector,
                relationship=data["relationship"]
            )

            voice_ids.append(voice_id)
            print(f"✓ 添加语音特征: {data['user_id']} - {data['person_name']} ({data['relationship']})")

        print(f"✓ 成功添加 {len(voice_ids)} 个语音特征")

        # 测试获取用户特征
        print("\n测试获取用户特征...")
        user_id = "user001"
        features = storage.get_user_all_features(user_id)

        print(f"用户 {user_id} 的特征分组: {list(features.keys())}")
        for person_key, vectors in features.items():
            print(f"  - {person_key}: {len(vectors)} 个特征向量")

        # 测试获取用户人员信息
        print("\n测试获取用户人员信息...")
        persons = storage.get_user_persons(user_id)

        print(f"用户 {user_id} 的人员信息:")
        for person in persons:
            print(f"  - 姓名: {person['person_name']}, 关系: {person['relationship']}, 音频数: {person['audio_count']}")

        # 测试用户统计
        print("\n测试用户统计...")
        stats = storage.get_user_stats(user_id)

        print(f"用户 {user_id} 统计信息:")
        print(f"  - 用户音频数量: {stats['user_audio_count']}")
        print(f"  - 总人员数: {stats['total_persons']}")
        print(f"  - 总音频特征数: {stats['total_audio_features']}")

        # 测试相似度搜索
        print("\n测试相似度搜索...")
        query_vector = np.random.rand(192)

        results = storage.find_most_similar(
            query_vector=query_vector,
            user_id=user_id,
            threshold=0.1,
            top_k=5
        )

        print(f"在用户 {user_id} 中搜索到 {len(results)} 个结果:")
        for result in results:
            print(f"  - {result['person_name']} ({result['relationship']}): 相似度 {result['similarity']:.3f}")

        # 测试存储信息
        print("\n测试存储信息...")
        info = storage.get_storage_info()

        print("存储信息:")
        print(f"  - 存储类型: {info['storage_type']}")
        print(f"  - 用户数量: {info['total_users']}")
        print(f"  - 基础目录: {info['base_directory']}")
        print(f"  - 每用户集合数: {info['collections_per_user']}")

        # 清理测试文件
        print("\n清理测试文件...")
        for file_path in test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✓ 删除测试文件: {file_path}")
            except Exception as e:
                print(f"✗ 删除文件失败 {file_path}: {e}")

        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)

        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """测试基本功能"""
    print("基本功能测试:")

    try:
        # 测试存储实例创建
        storage = get_multi_chroma_storage()
        print("✓ 存储实例创建成功")

        # 测试添加单个人物
        user_id = "test_user"
        person_name = "测试人员"
        relationship = "朋友"
        feature_vector = np.random.rand(192)

        voice_id = storage.add_voice_feature(
            user_id=user_id,
            person_name=person_name,
            feature_vector=feature_vector,
            relationship=relationship
        )

        print(f"✓ 添加语音特征成功: {voice_id}")

        # 测试获取特征
        features = storage.get_user_all_features(user_id)
        print(f"✓ 获取用户特征成功，找到 {len(features)} 个分组")

        # 测试搜索
        results = storage.find_most_similar(
            query_vector=feature_vector,
            user_id=user_id,
            threshold=0.1
        )

        print(f"✓ 相似度搜索成功，找到 {len(results)} 个结果")

        return True

    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False


def main():
    """主函数"""
    print("ChromaDB多用户集合测试开始...")

    # 运行基本功能测试
    print("\n1. 基本功能测试")
    basic_test = test_basic_functionality()

    # 运行完整功能测试
    print("\n2. 完整功能测试")
    full_test = test_chroma_storage()

    # 测试结果
    print(f"\n测试结果汇总:")
    print(f"  基本功能测试: {'通过' if basic_test else '失败'}")
    print(f"  完整功能测试: {'通过' if full_test else '失败'}")

    if basic_test and full_test:
        print("\n所有测试通过! ✓")
        return 0
    else:
        print("\n有测试失败 ✗")
        return 1


if __name__ == "__main__":
    exit(main())