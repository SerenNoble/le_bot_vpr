#!/usr/bin/env python3
"""
语音识别API启动脚本 - ChromaDB版本
"""
import os
import sys
import uvicorn
import argparse
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """打印启动横幅"""
    print("语音识别API服务 - ChromaDB版本")
    print("=" * 50)

def print_config(config: dict):
    """打印配置信息"""
    print(f"运行模式: ChromaDB")
    print(f"   服务地址: http://{config['host']}:{config['port']}")
    print(f"   调试模式: {config['debug']}")
    print(f"   缓存超时: {config['cache_timeout']}秒")
    print(f"   按需加载: {config['lazy_load']}")
    print("=" * 50)

def load_env_config() -> dict:
    """加载环境变量配置"""
    return {
        "host": os.getenv("API_HOST", "0.0.0.0"),
        "port": int(os.getenv("API_PORT", "8000")),
        "debug": os.getenv("API_DEBUG", "false").lower() == "true",
        "cache_timeout": int(os.getenv("CACHE_TIMEOUT", "300")),
        "lazy_load": os.getenv("LAZY_LOAD", "true").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "chroma_persist_dir": os.getenv("CHROMA_PERSIST_DIR", "./voice_chroma_db")
    }

def check_chroma_dependencies(config: dict) -> bool:
    """检查ChromaDB依赖"""
    try:
        import chromadb

        # 测试创建临时集合
        persist_dir = config["chroma_persist_dir"]
        test_client = chromadb.PersistentClient(path=persist_dir)

        # 尝试创建测试集合
        test_collection = test_client.get_or_create_collection("test_connection", metadata={"test": True})

        # 清理测试集合
        test_client.delete_collection("test_connection")

        print("ChromaDB环境检查通过")
        return True

    except ImportError:
        print("ChromaDB依赖未安装，请运行: pip install chromadb")
        return False
    except Exception as e:
        print(f"ChromaDB环境检查失败: {e}")
        return False

def check_model_files() -> bool:
    """检查模型文件是否存在"""
    try:
        from app.vpr import predictor

        # 尝试访问基础预测器
        if hasattr(predictor, 'model') or hasattr(predictor, '_extract_features'):
            print("语音特征提取模型检查通过")
            return True
        else:
            print("警告: 语音特征提取模型可能未正确加载")
            return True  # 不阻止启动，让用户在运行时发现问题

    except Exception as e:
        print(f"模型文件检查警告: {e}")
        return True  # 不阻止启动

def clear_chromadb_cache():
    """清理ChromaDB实例缓存"""
    try:
        print("清理ChromaDB实例缓存...")

        # 清理全局存储实例
        import app.db.multi_collection_chroma_storage as storage_module
        if hasattr(storage_module, '_multi_chroma_storage'):
            storage_module._multi_chroma_storage = None

        print("ChromaDB缓存清理完成")
    except Exception as e:
        print(f"清理ChromaDB缓存时出现警告: {e}")


def run_server(config: dict):
    """启动ChromaDB服务器"""
    print_config(config)

    try:
        # 清理可能存在的ChromaDB缓存
        clear_chromadb_cache()

        print("启动ChromaDB版本API服务...")
        app_module = "app.main:app"

        # 启动FastAPI应用
        uvicorn.run(
            app_module,
            host=config["host"],
            port=config["port"],
            reload=config["debug"],
            log_level=config["log_level"]
        )

    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print_banner()

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="语音识别API服务启动脚本 - ChromaDB版本")
    parser.add_argument(
        "--host",
        help="服务主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="服务端口 (默认: 8000)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查环境，不启动服务"
    )

    args = parser.parse_args()

    # 加载配置
    config = load_env_config()

    # 命令行参数覆盖配置
    if args.host:
        config["host"] = args.host
    if args.port:
        config["port"] = args.port
    if args.debug:
        config["debug"] = True

    print("运行模式: ChromaDB")

    # 环境检查
    print("\n环境检查...")

    # 检查ChromaDB依赖
    if not check_chroma_dependencies(config):
        print("请确保：")
        print("1. 安装ChromaDB: pip install chromadb")
        print("2. 检查文件系统权限")
        sys.exit(1)

    # 检查模型文件
    if not check_model_files():
        print("请确保：")
        print("1. 模型文件已下载到正确位置")
        print("2. 配置文件路径正确")

    if args.check_only:
        print("\n环境检查完成，ChromaDB系统正常")
        return

    print(f"\n准备启动ChromaDB模式服务...")

    # 启动服务器
    run_server(config)

if __name__ == "__main__":
    main()