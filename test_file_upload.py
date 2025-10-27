#!/usr/bin/env python3
"""
专门测试文件上传功能的脚本
"""
import requests
import os
import numpy as np
import wave

def create_simple_test_audio(filename="test_upload.wav", duration=3):
    """创建简单的测试音频文件"""
    sample_rate = 16000
    frequency = 440  # A4音符

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    audio_data += 0.1 * np.random.randn(len(t))  # 添加噪音
    audio_data = (audio_data * 32767).astype(np.int16)

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    print(f"✅ 创建测试音频: {filename}")
    return filename

def test_file_upload():
    """测试文件上传功能"""
    BASE_URL = "http://localhost:8000"

    print("🧪 测试文件上传功能")
    print("=" * 40)

    # 创建测试音频
    audio_file = create_simple_test_audio()

    try:
        # 测试1: 注册用户 - 文件上传
        print("\n1. 测试用户注册（文件上传）...")
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/api/v1/vpr/register/file?user_id=文件测试用户",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 注册成功: {result['message']}")
        else:
            print(f"❌ 注册失败: {response.status_code}")
            print(f"   错误信息: {response.json().get('detail', 'Unknown')}")
            return False

        # 测试2: 识别用户 - 文件上传
        print("\n2. 测试用户识别（文件上传）...")
        with open(audio_file, 'rb') as f:
            files = {'file': (os.path.basename(audio_file), f, 'audio/wav')}
            response = requests.post(
                f"{BASE_URL}/api/v1/vpr/recognize/file?threshold=0.6",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            if result['success'] and result['identified_user']:
                print(f"✅ 识别成功: {result['identified_user']}")
                print(f"   置信度: {result.get('confidence', 'N/A')}")
            else:
                print(f"⚠️  未识别到用户: {result['message']}")
        else:
            print(f"❌ 识别失败: {response.status_code}")
            print(f"   错误信息: {response.json().get('detail', 'Unknown')}")
            return False

        # 测试3: 删除用户
        print("\n3. 清理测试数据...")
        response = requests.delete(f"{BASE_URL}/api/v1/vpr/users/文件测试用户")
        if response.status_code == 200:
            print("✅ 用户删除成功")
        else:
            print(f"⚠️  用户删除失败: {response.status_code}")

        print("\n" + "=" * 40)
        print("🎉 文件上传功能测试完成！")
        return True

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保服务器已启动")
        print("   运行: python start_api.py")
        return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print(f"🧹 清理测试文件: {audio_file}")

def main():
    """主函数"""
    print("语音识别API - 文件上传功能测试")
    print("确保服务器已启动: python start_api.py")
    print()

    success = test_file_upload()

    if success:
        print("\n✅ 所有测试通过！")
        print("现在可以运行完整测试: python test_api.py")
    else:
        print("\n❌ 测试失败，请检查服务器状态")

if __name__ == "__main__":
    main()