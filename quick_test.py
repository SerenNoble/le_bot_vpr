#!/usr/bin/env python3
"""
快速测试脚本 - 测试API基本功能
"""
import requests
import time
import os

def test_api_basic():
    """基本API测试"""
    base_url = "http://localhost:8000"

    print("Quick API Test")
    print("=" * 30)

    # 1. 测试连接
    print("1. Testing connection...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("   OK - Server is running")
            print(f"   Message: {response.json().get('message', 'N/A')}")
        else:
            print(f"   FAIL - Status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   FAIL - Cannot connect to server")
        print("   Please start server first: python start_api.py")
        return False
    except Exception as e:
        print(f"   FAIL - Error: {e}")
        return False

    # 2. 测试获取用户列表
    print("\n2. Testing get users...")
    try:
        response = requests.get(f"{base_url}/api/v1/vpr/users", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"   OK - Users: {result['count']}")
            if result['users']:
                print(f"   List: {', '.join(result['users'][:3])}")  # 只显示前3个
        else:
            print(f"   FAIL - Status: {response.status_code}")
    except Exception as e:
        print(f"   FAIL - Error: {e}")

    # 3. 测试API根路径信息
    print("\n3. Getting API info...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            endpoints = response.json().get('endpoints', {})
            print("   Available endpoints:")
            for key, desc in endpoints.items():
                print(f"   - {desc}")
        else:
            print(f"   FAIL - Status: {response.status_code}")
    except Exception as e:
        print(f"   FAIL - Error: {e}")

    print("\n" + "=" * 30)
    print("Quick test completed!")
    print("For full API test, run: python test_api.py")
    print("For API docs, visit: http://localhost:8000/docs")

    return True

def main():
    """主函数"""
    print("Voice Recognition API - Quick Test")
    print("Make sure the server is running first!")
    print()

    # 等待用户确认或自动测试
    try:
        user_input = input("Press Enter to start test (or 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Test cancelled")
            return
    except KeyboardInterrupt:
        print("\nTest cancelled")
        return

    # 执行测试
    success = test_api_basic()

    if success:
        print("\nNext steps:")
        print("1. Create test audio: python create_test_audio.py")
        print("2. Run full test: python test_api.py")
        print("3. Try the API at: http://localhost:8000/docs")
    else:
        print("\nTroubleshooting:")
        print("1. Start the server: python start_api.py")
        print("2. Check if port 8000 is available")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()