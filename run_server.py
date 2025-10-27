#!/usr/bin/env python3
"""
语音识别API服务启动脚本
"""
import uvicorn
import os

if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault("PYTHONPATH", ".")

    # 启动FastAPI应用
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式下自动重载
        log_level="info"
    )