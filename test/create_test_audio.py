#!/usr/bin/env python3
"""
创建测试音频文件的工具
生成不同特征的测试音频用于语音识别测试
"""
import numpy as np
import wave
import os
from pathlib import Path

def generate_speech_like_audio(duration=5, sample_rate=16000,
                             fundamental_freq=150,
                             formants=[800, 1200, 2500, 3500],
                             noise_level=0.1):
    """
    生成类似语音的测试音频

    Args:
        duration: 音频时长（秒）
        sample_rate: 采样率
        fundamental_freq: 基频
        formants: 共振峰频率列表
        noise_level: 噪音水平

    Returns:
        numpy.ndarray: 音频数据
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 基频和泛音
    audio = np.zeros_like(t)
    for i in range(1, 6):  # 前5个泛音
        amplitude = 1.0 / i  # 泛音强度递减
        audio += amplitude * np.sin(2 * np.pi * fundamental_freq * i * t)

    # 添加共振峰（模拟元音）
    for formant_freq in formants:
        # 使用带通滤波器的简化版本
        bandwidth = 200
        envelope = np.exp(-((t * 1000 % 1000 - 500) ** 2) / (2 * bandwidth ** 2))
        audio += 0.3 * envelope * np.sin(2 * np.pi * formant_freq * t)

    # 添加包络（模拟语音强度变化）
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz的调制
    audio *= envelope

    # 添加呼吸噪音
    noise = noise_level * np.random.randn(len(t))
    # 对噪音进行低通滤波
    from scipy import signal
    b, a = signal.butter(4, 2000, fs=sample_rate, btype='low')
    if hasattr(signal, 'filtfilt'):
        noise = signal.filtfilt(b, a, noise)
    else:
        # 如果没有scipy，使用简单的移动平均
        noise = np.convolve(noise, np.ones(100)/100, mode='same')

    audio += noise

    # 归一化
    audio = audio / np.max(np.abs(audio)) * 0.8

    return audio

def create_test_audio_set():
    """创建一组测试音频文件"""

    # 创建测试目录
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)

    test_cases = [
        {
            "filename": "speaker1_male.wav",
            "params": {
                "duration": 4,
                "fundamental_freq": 120,  # 男性基频
                "formants": [700, 1100, 2400, 3400],  # 男性共振峰
                "noise_level": 0.08
            },
            "description": "模拟男性说话人1"
        },
        {
            "filename": "speaker2_female.wav",
            "params": {
                "duration": 5,
                "fundamental_freq": 200,  # 女性基频
                "formants": [850, 1350, 2600, 3600],  # 女性共振峰
                "noise_level": 0.06
            },
            "description": "模拟女性说话人1"
        },
        {
            "filename": "speaker1_male_2.wav",
            "params": {
                "duration": 3,
                "fundamental_freq": 125,  # 稍有不同的男性基频
                "formants": [720, 1120, 2420, 3420],
                "noise_level": 0.09
            },
            "description": "模拟男性说话人1（不同样本）"
        },
        {
            "filename": "speaker2_female_2.wav",
            "params": {
                "duration": 4,
                "fundamental_freq": 195,  # 稍有不同的女性基频
                "formants": [860, 1360, 2620, 3620],
                "noise_level": 0.07
            },
            "description": "模拟女性说话人1（不同样本）"
        },
        {
            "filename": "speaker3_male.wav",
            "params": {
                "duration": 6,
                "fundamental_freq": 110,  # 另一个男性基频
                "formants": [680, 1080, 2380, 3380],
                "noise_level": 0.1
            },
            "description": "模拟男性说话人2"
        }
    ]

    print("🎵 创建测试音频文件...")

    for i, case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 创建: {case['description']}")

        try:
            # 生成音频数据
            audio_data = generate_speech_like_audio(**case['params'])

            # 转换为16位整数
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # 保存为WAV文件
            filepath = test_dir / case['filename']
            with wave.open(str(filepath), 'w') as wav_file:
                wav_file.setnchannels(1)  # 单声道
                wav_file.setsampwidth(2)  # 16位
                wav_file.setframerate(16000)  # 16kHz采样率
                wav_file.writeframes(audio_int16.tobytes())

            print(f"   ✅ 已保存: {filepath}")

        except Exception as e:
            print(f"   ❌ 创建失败: {e}")

    print(f"\n📁 测试音频文件已保存到: {test_dir.absolute()}")
    print("\n📋 文件说明:")
    for case in test_cases:
        print(f"   • {case['filename']}: {case['description']}")

    return test_dir

def create_simple_test_audio():
    """创建简单的测试音频（仅使用numpy）"""
    print("🎵 创建简单测试音频...")

    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)

    # 生成简单的正弦波
    duration = 3
    sample_rate = 16000
    frequency = 440  # A4音符

    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # 创建两个不同的测试音频
    audio1 = np.sin(2 * np.pi * frequency * t)
    audio2 = np.sin(2 * np.pi * (frequency + 100) * t)  # 不同频率

    # 添加一些包络和噪音
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)
    audio1 *= envelope
    audio2 *= envelope

    audio1 += 0.05 * np.random.randn(len(t))
    audio2 += 0.05 * np.random.randn(len(t))

    # 归一化
    audio1 = audio1 / np.max(np.abs(audio1)) * 0.8
    audio2 = audio2 / np.max(np.abs(audio2)) * 0.8

    # 转换为16位整数
    audio1_int16 = (audio1 * 32767).astype(np.int16)
    audio2_int16 = (audio2 * 32767).astype(np.int16)

    # 保存文件
    files = [
        ("simple_test_1.wav", audio1_int16),
        ("simple_test_2.wav", audio2_int16)
    ]

    for filename, audio_data in files:
        filepath = test_dir / filename
        with wave.open(str(filepath), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        print(f"   ✅ 已保存: {filepath}")

    return test_dir

def main():
    """主函数"""
    print("🎤 语音识别测试音频生成工具")
    print("=" * 50)

    try:
        # 尝试导入scipy
        import scipy.signal
        print("✅ 检测到scipy，将生成高质量语音模拟音频")
        test_dir = create_test_audio_set()
    except ImportError:
        print("⚠️  未检测到scipy，将生成简单测试音频")
        print("   如需高质量音频，请安装: pip install scipy")
        test_dir = create_simple_test_audio()

    print(f"\n🎯 测试建议:")
    print(f"   1. 启动API服务: python run_server.py")
    print(f"   2. 运行测试脚本: python test_api.py")
    print(f"   3. 或手动测试:")
    print(f"      - 注册用户: 使用 {test_dir}/speaker1_male.wav")
    print(f"      - 识别测试: 使用 {test_dir}/speaker1_male_2.wav")
    print(f"      - 对比测试: 使用 {test_dir}/speaker1_male.wav 和 {test_dir}/speaker2_female.wav")

    print(f"\n📂 测试文件位置: {test_dir.absolute()}")

if __name__ == "__main__":
    main()