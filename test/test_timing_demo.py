"""
高精度计时功能演示
展示predictor各方法的执行时间统计
"""
import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from app.vpr import predictor


def demo_timing():
    """演示计时功能"""

    print("\n" + "="*80)
    print("🎯 高精度计时功能演示")
    print("="*80)

    # 测试文件路径
    audio_file1 = os.path.join(project_root, "test_audio_1.wav")
    audio_file2 = os.path.join(project_root, "test_audio_2.wav")
    market_audio = os.path.join(project_root, "market_test_500ms.wav")

    print("\n1️⃣  测试 get_users() 方法...")
    users = predictor.get_users()
    print(f"   找到 {len(users)} 个用户")

    print("\n2️⃣  测试 predict() 方法...")
    if os.path.exists(audio_file1):
        feature = predictor.predict(audio_file1)
        print(f"   提取到 {feature.shape[0]} 维特征向量")
    else:
        print(f"   ⚠️  文件不存在: {audio_file1}")

    print("\n3️⃣  测试 contrast() 方法...")
    if os.path.exists(audio_file1) and os.path.exists(audio_file2):
        similarity = predictor.contrast(audio_file1, audio_file2)
        print(f"   相似度: {similarity:.5f}")
    else:
        print(f"   ⚠️  对比文件不存在")

    print("\n4️⃣  测试 recognition() 方法...")
    if os.path.exists(market_audio):
        user_name, similarity = predictor.recognition(market_audio)
        if user_name:
            print(f"   识别到用户: {user_name}, 相似度: {similarity}")
        else:
            print(f"   未识别到用户")
    else:
        print(f"   ⚠️  文件不存在: {market_audio}")

    print("\n5️⃣  再次测试 recognition() 以收集多次调用数据...")
    if os.path.exists(audio_file1):
        for i in range(3):
            predictor.recognition(audio_file1)
            print(f"   第 {i+1} 次调用完成")

    # 打印统计信息
    predictor.print_timing_stats()

    # 展示如何获取原始统计数据
    print("\n" + "="*80)
    print("📊 获取原始统计数据示例")
    print("="*80)
    stats = predictor.get_timing_stats()
    for method_name, data in stats.items():
        print(f"\n{method_name}:")
        print(f"  调用次数: {data['count']}")
        print(f"  总耗时: {data['total_time']*1000:.2f} ms")
        print(f"  平均耗时: {(data['total_time']/data['count'])*1000:.2f} ms")
        print(f"  最小耗时: {data['min_time']*1000:.2f} ms")
        print(f"  最大耗时: {data['max_time']*1000:.2f} ms")
        print(f"  最后一次: {data['last_time']*1000:.2f} ms")

    # 演示重置统计
    print("\n" + "="*80)
    print("🔄 重置统计数据并再次测试...")
    print("="*80)
    predictor.reset_timing_stats()

    if os.path.exists(audio_file1):
        predictor.predict(audio_file1)
        predictor.recognition(audio_file1)

    predictor.print_timing_stats()


if __name__ == "__main__":
    demo_timing()

