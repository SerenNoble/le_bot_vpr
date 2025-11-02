"""
本地测试文件 - 直接使用predictor访问本地音频文件
不使用任何服务器或网络传输
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from app.vpr import predictor


def test_register_user():
    """测试注册用户"""
    print("\n" + "="*60)
    print("测试1: 注册用户")
    print("="*60)

    # 使用已有的测试音频文件
    audio_file = os.path.join(project_root, "test_audio_1.wav")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return False

    print(f"📁 使用音频文件: {audio_file}")

    # 注册用户
    user_name = "本地测试用户1"
    success, message = predictor.register(audio_file, user_name)

    if success:
        print(f"✅ {message} - 用户名: {user_name}")
        return True
    else:
        print(f"❌ 注册失败: {message}")
        return False


def test_recognition():
    """测试声纹识别"""
    print("\n" + "="*60)
    print("测试2: 声纹识别")
    print("="*60)

    # 使用测试音频进行识别
    audio_file = os.path.join(project_root, "test_audio_1.wav")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return False

    print(f"📁 使用音频文件: {audio_file}")

    # 识别用户
    result = predictor.recognition(audio_file)
    user_name, similarity = result

    if user_name:
        print(f"✅ 识别成功!")
        print(f"   用户名: {user_name}")
        print(f"   相似度: {similarity}")
        return True
    else:
        print(f"❌ 未识别到用户")
        return False


def test_contrast():
    """测试声纹对比"""
    print("\n" + "="*60)
    print("测试3: 声纹对比")
    print("="*60)

    # 使用两个不同的音频文件进行对比
    audio_file1 = os.path.join(project_root, "test_audio_1.wav")
    audio_file2 = os.path.join(project_root, "test_audio_2.wav")

    if not os.path.exists(audio_file1):
        print(f"❌ 音频文件1不存在: {audio_file1}")
        return False

    if not os.path.exists(audio_file2):
        print(f"❌ 音频文件2不存在: {audio_file2}")
        return False

    print(f"📁 音频文件1: {audio_file1}")
    print(f"📁 音频文件2: {audio_file2}")

    # 对比两个音频
    similarity = predictor.contrast(audio_file1, audio_file2)

    print(f"📊 相似度: {similarity:.5f}")

    if similarity >= predictor.threshold:
        print(f"✅ 两个音频是同一个人 (阈值: {predictor.threshold})")
    else:
        print(f"❌ 两个音频不是同一个人 (阈值: {predictor.threshold})")

    return True


def test_predict_feature():
    """测试提取音频特征"""
    print("\n" + "="*60)
    print("测试4: 提取音频特征向量")
    print("="*60)

    audio_file = os.path.join(project_root, "test_audio_1.wav")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return False

    print(f"📁 使用音频文件: {audio_file}")

    # 提取特征
    feature = predictor.predict(audio_file)

    print(f"✅ 成功提取特征向量")
    print(f"   特征维度: {feature.shape}")
    print(f"   特征前10个值: {feature[:10]}")

    return True


def test_get_users():
    """测试获取所有用户"""
    print("\n" + "="*60)
    print("测试5: 获取所有注册用户")
    print("="*60)

    users = predictor.get_users()

    print(f"📋 共有 {len(users)} 个用户:")
    for i, user in enumerate(users, 1):
        print(f"   {i}. {user}")

    return True


def test_audio_db_users():
    """测试音频数据库中的用户识别"""
    print("\n" + "="*60)
    print("测试6: 测试音频数据库中的用户")
    print("="*60)

    audio_db_path = os.path.join(project_root, "audio_db")

    if not os.path.exists(audio_db_path):
        print(f"❌ 音频数据库路径不存在: {audio_db_path}")
        return False

    # 遍历音频数据库中的所有用户
    for user_name in os.listdir(audio_db_path):
        user_dir = os.path.join(audio_db_path, user_name)
        if not os.path.isdir(user_dir):
            continue

        print(f"\n👤 测试用户: {user_name}")

        # 获取该用户的第一个音频文件
        audio_files = [f for f in os.listdir(user_dir) if f.endswith('.wav')]
        if not audio_files:
            print(f"   ⚠️  该用户没有音频文件")
            continue

        audio_file = os.path.join(user_dir, audio_files[0])
        print(f"   📁 音频文件: {audio_file}")

        # 进行识别
        result = predictor.recognition(audio_file)
        recognized_name, similarity = result

        if recognized_name:
            if recognized_name == user_name:
                print(f"   ✅ 正确识别: {recognized_name} (相似度: {similarity})")
            else:
                print(f"   ❌ 识别错误: 识别为 {recognized_name} (相似度: {similarity})")
        else:
            print(f"   ❌ 未能识别")

    return True


def test_market_audio():
    """测试市场测试音频"""
    print("\n" + "="*60)
    print("测试7: 测试市场测试音频")
    print("="*60)

    audio_file = os.path.join(project_root, "market_test_500ms.wav")

    if not os.path.exists(audio_file):
        print(f"❌ 音频文件不存在: {audio_file}")
        return False

    print(f"📁 使用音频文件: {audio_file}")

    # 进行识别
    result = predictor.recognition(audio_file)
    user_name, similarity = result

    if user_name:
        print(f"✅ 识别成功!")
        print(f"   用户名: {user_name}")
        print(f"   相似度: {similarity}")
    else:
        print(f"❌ 未识别到用户")

    return True


def main():
    """主测试函数"""
    print("\n")
    print("🎯 声纹识别本地测试")
    print("="*60)
    print(f"项目根目录: {project_root}")
    print(f"阈值: {predictor.threshold}")
    print("="*60)

    # 运行所有测试
    tests = [
        ("获取所有用户", test_get_users),
        ("提取音频特征向量", test_predict_feature),
        ("声纹对比", test_contrast),
        ("注册用户", test_register_user),
        ("声纹识别", test_recognition),
        ("音频数据库用户测试", test_audio_db_users),
        ("市场测试音频", test_market_audio),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{test_name}' 发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 打印测试总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")
    print("="*60)

    # 打印高精度计时统计
    print("\n")
    predictor.print_timing_stats()


if __name__ == "__main__":
    main()

