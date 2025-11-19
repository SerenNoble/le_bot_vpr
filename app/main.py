"""
语音识别API - 多用户集合ChromaDB版本
基于多用户集合ChromaDB向量数据库的高性能声纹识别系统
每个用户独立集合，提供更好的数据隔离和查询性能
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
import os
import time
from typing import Optional, List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入ChromaDB预测器和模型
from app.vpr.chroma_predictor import chroma_predictor
from app.models.chroma_models import (
    RegisterResponse, RecognitionResponse, PersonInfo, UserInfo,
    UserListResponse, StatsResponse
)

app = FastAPI(
    title="Voice Recognition API - Multi-Collection ChromaDB Version",
    description="高性能语音识别API - 基于多用户集合ChromaDB向量数据库\n每个用户独立集合，提供更好的数据隔离和查询性能",
    version="5.0.0"
)

# 请求/响应模型
class RegisterRequest(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    person_name: str = Field(..., description="人员姓名（实际姓名，如：张三、李四等）")
    relationship: str = Field("朋友", description="与用户的关系（如：本人、爸爸、妈妈、朋友、同事等）")

class RecognitionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="指定用户ID（如果不指定则在所有用户中搜索）")
    threshold: Optional[float] = Field(0.6, description="识别阈值", ge=0.0, le=1.0)

# 生命周期事件
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化系统"""
    try:
        await chroma_predictor.initialize()
        logger.info("✅ 应用启动完成，ChromaDB连接正常")
    except Exception as e:
        logger.error(f"❌ 应用启动失败: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    try:
        logger.info("✅ 应用关闭完成，资源已清理")
    except Exception as e:
        logger.error(f"❌ 应用关闭时出错: {e}")

# API端点
@app.post("/api/v4/vpr/register", response_model=RegisterResponse)
async def register_user_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    person_name: str = Form(...),
    relationship: str = Form("朋友")
):
    """
    注册用户音频特征
    支持为用户添加多个人员的音频

    Args:
        file: 音频文件
        user_id: 用户唯一标识
        person_name: 人员姓名（实际姓名，如：张三、李四等）
        relationship: 与用户的关系（如：本人、爸爸、妈妈、朋友、同事等）
    """
    try:
        # 参数验证
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id 不能为空")
        if not person_name:
            raise HTTPException(status_code=400, detail="person_name 不能为空")

        # 检查文件格式
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}"
            )

        # 读取音频数据
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")

        # 注册音频特征
        start_time = time.time()
        success, message = await chroma_predictor.register_user_voice(
            user_id=user_id,
            person_name=person_name,
            audio_data=audio_bytes,
            relationship=relationship
        )
        processing_time = (time.time() - start_time) * 1000

        if success:
            logger.info(f"✅ 注册成功: 用户={user_id}, 人员={person_name}, 耗时={processing_time:.2f}ms")
            return RegisterResponse(
                success=True,
                message=message,
                user_id=user_id,
                person_name=person_name,
                registration_time=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        else:
            raise HTTPException(status_code=400, detail=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 注册音频失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频文件失败: {str(e)}")

@app.post("/api/v4/vpr/recognize", response_model=RecognitionResponse)
async def identify_user_audio(
    file: UploadFile = File(...),
    user_id: str = Form(None),
    threshold: float = Form(0.6)
):
    """
    识别用户音频
    可以指定在特定用户下搜索，或在所有用户中搜索
    """
    try:
        # 参数验证
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(status_code=400, detail="阈值必须在0.0到1.0之间")

        # 检查文件格式
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}"
            )

        # 读取音频数据
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="音频文件为空")

        # 识别音频
        result = await chroma_predictor.recognize_user_voice(
            audio_data=audio_bytes,
            threshold=threshold,
            target_user_id=user_id
        )

        logger.info(f"🔍 识别完成: 成功={result.success}, 耗时={result.processing_time_ms:.2f}ms")

        if result.success:
            person_type = "本人" if result.is_user else "人员"
            return RecognitionResponse(
                success=True,
                message=f"成功识别: {result.person_name} ({person_type})",
                user_id=result.user_id,
                voice_id=result.voice_id,
                person_id=result.person_id,
                person_name=result.person_name,
                is_user=result.is_user,
                confidence=result.confidence,
                similarity=result.similarity,
                processing_time_ms=result.processing_time_ms,
                match_details=result.match_details
            )
        else:
            return RecognitionResponse(
                success=False,
                message="未能识别出匹配的用户",
                confidence=0.0,
                processing_time_ms=result.processing_time_ms,
                match_details=result.match_details
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 识别音频失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频文件失败: {str(e)}")

@app.get("/api/v4/vpr/users", response_model=UserListResponse)
async def get_registered_users():
    """
    获取所有已注册的用户及其人员信息
    """
    try:
        users = await chroma_predictor.search_users(limit=100)

        return UserListResponse(
            success=True,
            users=[
                UserInfo(
                    user_id=user["user_id"],
                    user_name=user.get("user_name"),
                    total_persons=user["total_persons"],
                    total_audio_features=user["user_audio_count"],
                    persons=[]  # 简化版不返回详细人员列表
                )
                for user in users
            ],
            count=len(users)
        )
    except Exception as e:
        logger.error(f"❌ 获取用户列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")

@app.get("/api/v4/vpr/users/{user_id}/persons", response_model=List[PersonInfo])
async def get_user_persons(user_id: str):
    """
    获取指定用户的所有人员信息
    """
    try:
        persons = await chroma_predictor.get_user_persons(user_id)
        return [PersonInfo(**person) for person in persons]
    except Exception as e:
        logger.error(f"❌ 获取用户人员失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户人员失败: {str(e)}")

@app.delete("/api/v4/vpr/users/{user_id}")
async def delete_user(user_id: str):
    """
    删除指定用户及其所有数据
    """
    try:
        success, message = await chroma_predictor.delete_user(user_id)
        if success:
            return {"success": True, "message": message}
        else:
            raise HTTPException(status_code=404, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 删除用户失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除用户失败: {str(e)}")

@app.delete("/api/v4/vpr/users/{user_id}/persons/{person_id}")
async def delete_person(user_id: str, person_id: str):
    """
    删除用户下的特定人员及其所有音频
    """
    try:
        success, message = await chroma_predictor.delete_person(user_id, person_id)
        if success:
            return {"success": True, "message": message}
        else:
            raise HTTPException(status_code=404, detail=message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 删除人员失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除人员失败: {str(e)}")

@app.get("/api/v4/vpr/stats/{user_id}", response_model=StatsResponse)
async def get_user_stats(user_id: str):
    """
    获取指定用户的统计信息
    """
    try:
        stats = await chroma_predictor.get_user_stats(user_id)
        return StatsResponse(
            success=True,
            stats=stats,
            message="用户统计信息获取成功"
        )
    except Exception as e:
        logger.error(f"❌ 获取用户统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取用户统计失败: {str(e)}")

@app.get("/api/v4/vpr/stats", response_model=StatsResponse)
async def get_global_stats():
    """
    获取全局统计信息
    """
    try:
        stats = await chroma_predictor.get_user_stats()
        return StatsResponse(
            success=True,
            stats=stats,
            message="全局统计信息获取成功"
        )
    except Exception as e:
        logger.error(f"❌ 获取全局统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取全局统计失败: {str(e)}")

@app.post("/api/v4/vpr/cache/clear")
async def clear_cache():
    """
    清空内存缓存
    """
    try:
        await chroma_predictor.clear_cache()
        return {
            "success": True,
            "message": "内存缓存已清空"
        }
    except Exception as e:
        logger.error(f"❌ 清空缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空缓存失败: {str(e)}")

@app.get("/api/v4/vpr/storage/info")
async def get_storage_info():
    """
    获取存储信息
    """
    try:
        info = await chroma_predictor.get_storage_info()
        return {
            "success": True,
            "storage_info": info
        }
    except Exception as e:
        logger.error(f"❌ 获取存储信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取存储信息失败: {str(e)}")

@app.post("/api/v4/vpr/storage/clear")
async def clear_all_data():
    """
    清空所有数据（危险操作）
    """
    try:
        success = await chroma_predictor.clear_all_data()
        if success:
            return {
                "success": True,
                "message": "所有语音特征数据已清空"
            }
        else:
            return {
                "success": False,
                "message": "清空数据失败"
            }
    except Exception as e:
        logger.error(f"❌ 清空数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空数据失败: {str(e)}")

@app.get("/")
async def root():
    """
    API根路径
    """
    storage_info = await chroma_predictor.get_storage_info()

    return {
        "message": "高性能语音识别API服务 - ChromaDB版本",
        "version": "4.0.0",
        "features": [
            "ChromaDB向量数据库",
            "高性能向量检索",
            "支持多用户管理",
            "智能缓存机制",
            "本地文件存储",
            "毫秒级响应"
        ],
        "storage_info": storage_info,
        "endpoints": {
            "register": "/api/v4/vpr/register - 注册用户音频(文件上传)",
            "recognize": "/api/v4/vpr/recognize - 识别用户音频(文件上传)",
            "users": "/api/v4/vpr/users - 获取所有用户",
            "user_persons": "/api/v4/vpr/users/{user_id}/persons - 获取用户的人员列表",
            "delete_user": "/api/v4/vpr/users/{user_id} - 删除用户",
            "delete_person": "/api/v4/vpr/users/{user_id}/persons/{person_id} - 删除人员",
            "user_stats": "/api/v4/vpr/stats/{user_id} - 获取用户统计",
            "global_stats": "/api/v4/vpr/stats - 获取全局统计",
            "storage_info": "/api/v4/vpr/storage/info - 获取存储信息",
            "clear_cache": "/api/v4/vpr/cache/clear - 清空缓存",
            "clear_data": "/api/v4/vpr/storage/clear - 清空所有数据",
            "docs": "/docs - API文档"
        }
    }

@app.get("/health")
async def health_check():
    """
    健康检查接口
    """
    try:
        storage_info = await chroma_predictor.get_storage_info()

        return {
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "database": "connected",
            "database_type": "ChromaDB",
            "storage_info": storage_info,
            "version": "4.0.0"
        }
    except Exception as e:
        logger.error(f"❌ 健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
            "version": "4.0.0"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)