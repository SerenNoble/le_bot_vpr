#!/usr/bin/env python3
"""
语音识别API测试脚本
测试所有API接口的功能
"""
import requests
import base64
import json
import time
import os
import time
from pathlib import Path

# API基础URL
BASE_URL = "http://localhost:8000"

def create_test_audio_file(filename="test_audio.wav", duration=3):
    """创建测试用的音频文件"""
    try:
        import numpy as np
        import wave

        # 音频参数
        sample_rate = 16000
        frequency = 440  # A4音符

        # 生成正弦波
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # 添加一些噪音和变化使其更像真实语音
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)  # 添加泛音
        audio_data += 0.05 * np.random.randn(len(t))  # 添加噪音
        audio_data = (audio_data * 32767).astype(np.int16)

        # 保存为WAV文件
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"✅ 创建测试音频文件: {filename}")
        return filename

    except ImportError:
        print("⚠️  需要安装numpy和wave库来创建测试音频文件")
        print("   pip install numpy")
        return None

def audio_to_base64(audio_file):
    """将音频文件转换为base64"""
    try:
        with open(audio_file, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ 音频文件不存在: {audio_file}")
        return None

def test_connection():
    """测试API连接"""
    print("🔗 测试API连接...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API服务连接成功")
            print(f"   服务信息: {response.json()['message']}")
            return True
        else:
            print(f"❌ API服务连接失败: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务，请确保服务已启动")
        return False

def test_register_user_base64(user_id, audio_file):
    """测试base64方式注册用户"""
    print(f"\n🎤 测试注册用户 (Base64): {user_id}")

    audio_base64 = audio_to_base64(audio_file)
    if not audio_base64:
        return False

    data = {
        "data": audio_base64,
        "format": "wav",
        "codec": "pcm"
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/vpr/register?user_id={user_id}", json=data)
        result = response.json()

        if response.status_code == 200:
            print(f"✅ 用户注册成功: {result['message']}")
            print(f"   特征形状: {result.get('feature_shape', 'N/A')}")
            return True
        else:
            print(f"❌ 用户注册失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_register_user_file(user_id, audio_file):
    """测试文件上传方式注册用户"""
    print(f"\n📁 测试注册用户 (文件上传): {user_id}")

    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/api/v1/vpr/register/file?user_id={user_id}",
                files=files
            )

        result = response.json()

        if response.status_code == 200:
            print(f"✅ 用户注册成功: {result['message']}")
            print(f"   特征形状: {result.get('feature_shape', 'N/A')}")
            return True
        else:
            print(f"❌ 用户注册失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 注册请求异常: {e}")
        return False

def test_recognize_user_base64(audio_file):
    """测试base64方式识别用户"""
    print(f"\n🔍 测试用户识别 (Base64)")

    audio_base64 = audio_to_base64(audio_file)
    if not audio_base64:
        return False

    data = {
        "data": audio_base64,
        "format": "wav",
        "codec": "pcm"
    }

    try:

        response = requests.post(f"{BASE_URL}/api/v1/vpr/recognize?threshold=0.6", json=data)
        result = response.json()

        if response.status_code == 200:
            if result['success'] and result['identified_user']:
                print(f"✅ 识别成功: {result['identified_user']}")
                print(f"   置信度: {result.get('confidence', 'N/A')}")
            else:
                print(f"⚠️  未识别到匹配用户: {result['message']}")
            return True
        else:
            print(f"❌ 用户识别失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 识别请求异常: {e}")
        return False

def test_recognize_user_file(audio_file):
    """测试文件上传方式识别用户"""
    print(f"\n📁 测试用户识别 (文件上传)")

    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/api/v1/vpr/recognize/file?threshold=0.6",
                files=files
            )
            print(f"   耗时: {(time.time() - start_time) * 1000:.2f} ms")

        result = response.json()
        # ms统计


        if response.status_code == 200:
            if result['success'] and result['identified_user']:
                print(f"✅ 识别成功: {result['identified_user']}")
                print(f"   置信度: {result.get('confidence', 'N/A')}")
            else:
                print(f"⚠️  未识别到匹配用户: {result['message']}")
            return True
        else:
            print(f"❌ 用户识别失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 识别请求异常: {e}")
        return False

def test_get_users():
    """测试获取用户列表"""
    print(f"\n👥 测试获取用户列表")

    try:

        response = requests.get(f"{BASE_URL}/api/v1/vpr/users")
        result = response.json()

        if response.status_code == 200:
            print(f"✅ 获取用户列表成功")
            print(f"   用户数量: {result['count']}")
            if result['users']:
                print(f"   用户列表: {', '.join(result['users'])}")
            else:
                print("   暂无注册用户")
            return True
        else:
            print(f"❌ 获取用户列表失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 获取用户列表异常: {e}")
        return False

def test_compare_voices_base64(audio_file1, audio_file2):
    """测试base64方式语音对比"""
    print(f"\n🔄 测试语音对比 (Base64)")

    audio_base64_1 = audio_to_base64(audio_file1)
    audio_base64_2 = audio_to_base64(audio_file2)

    if not audio_base64_1 or not audio_base64_2:
        return False

    data = {
        "audio_data1": {
            "data": audio_base64_1,
            "format": "wav",
            "codec": "pcm"
        },
        "audio_data2": {
            "data": audio_base64_2,
            "format": "wav",
            "codec": "pcm"
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/v1/vpr/compare?threshold=0.6", json=data)
        result = response.json()

        if response.status_code == 200:
            print(f"✅ 语音对比完成")
            print(f"   相似度: {result['similarity']:.4f}")
            print(f"   是否为同一人: {'是' if result['is_same_person'] else '否'}")
            return True
        else:
            print(f"❌ 语音对比失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 语音对比异常: {e}")
        return False

def test_delete_user(user_id):
    """测试删除用户"""
    print(f"\n🗑️  测试删除用户: {user_id}")

    try:
        response = requests.delete(f"{BASE_URL}/api/v1/vpr/users/{user_id}")
        result = response.json()

        if response.status_code == 200:
            print(f"✅ 用户删除成功: {result['message']}")
            return True
        else:
            print(f"❌ 用户删除失败: {result.get('detail', '未知错误')}")
            return False
    except Exception as e:
        print(f"❌ 删除用户异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始语音识别API测试")
    print("=" * 50)

    # 测试连接
    if not test_connection():
        print("\n❌ 请先启动API服务：")
        print("   python run_server.py")
        return

    # 创建测试音频文件
    print("\n🎵 创建测试音频文件...")
    audio_file1 = create_test_audio_file("test_audio_1.wav", duration=3)
    audio_file2 = create_test_audio_file("test_audio_2.wav", duration=4)

    if not audio_file1 or not audio_file2:
        print("\n❌ 无法创建测试音频文件，测试终止")
        return
    test_get_users()
    test_register_user_base64("测试用户1", audio_file1)
    test_recognize_user_file(audio_file1)
    exit()
    # 测试用例
    test_cases = [
        # 1. 获取初始用户列表
        ("获取用户列表", lambda: test_get_users()),

        # 2. 注册用户 - Base64方式
        ("注册用户-Base64", lambda: test_register_user_base64("测试用户1", audio_file1)),

        # 3. 注册用户 - 文件上传方式
        ("注册用户-文件上传", lambda: test_register_user_file("测试用户2", audio_file2)),

        # 4. 获取更新后的用户列表
        ("获取用户列表", lambda: test_get_users()),

        # 5. 识别用户 - Base64方式
        ("识别用户-Base64", lambda: test_recognize_user_base64(audio_file1)),

        # 6. 识别用户 - 文件上传方式
        ("识别用户-文件上传", lambda: test_recognize_user_file(audio_file2)),

        # 8. 删除用户
        ("删除用户1", lambda: test_delete_user("测试用户1")),
        ("删除用户2", lambda: test_delete_user("测试用户2")),

        # 9. 获取最终用户列表
        ("获取用户列表", lambda: test_get_users()),
    ]

    # 执行测试
    passed = 0
    total = len(test_cases)

    for test_name, test_func in test_cases:
        try:
            if test_func():
                passed += 1
            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            print(f"❌ 测试 '{test_name}' 异常: {e}")

    # 清理测试文件
    print(f"\n🧹 清理测试文件...")
    for file in [audio_file1, audio_file2]:
        if file and os.path.exists(file):
            os.remove(file)
            print(f"   删除: {file}")

    # 测试结果
    print("\n" + "=" * 50)
    print(f"📊 测试完成: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！API服务运行正常")
    else:
        print("⚠️  部分测试失败，请检查API服务")

    print("\n💡 提示:")
    print("   - 如需查看详细API文档，访问: http://localhost:8000/docs")
    print("   - 如需调整识别阈值，修改请求中的threshold参数")
    print("   - 建议使用真实语音文件进行测试以获得更好效果")

if __name__ == "__main__":
    main()