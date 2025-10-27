#!/usr/bin/env python3
"""
简化的API启动脚本
避免Windows控制台编码问题
"""
import os
import sys
import uvicorn

def main():
    print("Starting Voice Recognition API Server...")
    print("=" * 50)

    # 设置环境变量
    os.environ.setdefault("PYTHONPATH", ".")

    # 检查必要文件
    config_file = "app/vpr/configs/eres2net.yml"
    model_dir = "app/vpr/models/ERes2Net_Fbank/best_model"

    if not os.path.exists(config_file):
        print(f"ERROR: Config file not found: {config_file}")
        return 1

    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    print("Files checked successfully!")
    print("Starting server at http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # 启动FastAPI应用
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # 避免重载时的路径问题
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"Server error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())