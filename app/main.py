from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import base64
import binascii
import os
from typing import Optional, List

from app.vpr import predictor

app = FastAPI(
    title="Voice Recognition API",
    description="语音识别和注册API",
    version="1.0.0"
)


class AudioData(BaseModel):
    data: str
    sample_rate: int = 16000
    format: str = "wav"
    codec: str = "pcm"


class RegisterResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    feature_shape: Optional[str] = None


class RecognitionResponse(BaseModel):
    success: bool
    message: str
    identified_user: Optional[str] = None
    confidence: Optional[float] = None
    feature_shape: Optional[str] = None


class UserListResponse(BaseModel):
    success: bool
    users: List[str]
    count: int


class DeleteUserResponse(BaseModel):
    success: bool
    message: str


class CompareResponse(BaseModel):
    success: bool
    message: str
    similarity: float
    is_same_person: bool


@app.post("/api/v1/vpr/register", response_model=RegisterResponse)
async def register_user_audio(audio_data: AudioData, user_id: str):
    """
    注册用户音频特征
    """
    try:
        # 解码base64音频数据为字节
        audio_bytes = base64.b64decode(audio_data.data)

        # 获取音频特征
        embedding = predictor.predict(audio_data=audio_bytes)

        # 注册用户音频
        success, message = predictor.register(user_name=user_id, audio_data=audio_bytes)

        if success:
            return RegisterResponse(
                success=True,
                message=f"用户 {user_id} 注册成功",
                user_id=user_id,
                feature_shape=str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )
        else:
            raise HTTPException(status_code=400, detail=f"注册失败: {message}")

    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"无效的base64编码: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}")


@app.post("/api/v1/vpr/register/file", response_model=RegisterResponse)
async def register_user_audio_file(file: UploadFile = File(...), user_id: str = ""):
    """
    通过上传文件注册用户音频特征
    """
    try:
        # 检查文件格式 - 更宽松的检查
        if file.content_type and not file.content_type.startswith('audio/'):
            # 如果明确指定了非audio类型才拒绝
            if file.content_type not in ['application/octet-stream', 'application/x-download']:
                raise HTTPException(status_code=400, detail="只支持音频文件")

        # 检查文件扩展名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}")

        # 读取音频数据
        audio_bytes = await file.read()

        # 获取音频特征
        embedding = predictor.predict(audio_data=audio_bytes)

        # 注册用户音频
        success, message = predictor.register(user_name=user_id or file.filename, audio_data=audio_bytes)

        if success:
            return RegisterResponse(
                success=True,
                message=f"用户 {user_id or file.filename} 注册成功",
                user_id=user_id or file.filename,
                feature_shape=str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )
        else:
            raise HTTPException(status_code=400, detail=f"注册失败: {message}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理音频文件失败: {str(e)}")

@app.post("/api/v1/vpr/recognize", response_model=RecognitionResponse)
async def identify_user(audio_data: AudioData, threshold: Optional[float] = None):
    """
    识别用户音频
    """
    try:
        # 解码base64音频数据为字节
        audio_bytes = base64.b64decode(audio_data.data)

        # 获取音频特征
        embedding = predictor.predict(audio_data=audio_bytes)

        # 识别用户
        result = predictor.recognition(audio_data=audio_bytes, threshold=threshold)

        if result[0] is not None:
            return RecognitionResponse(
                success=True,
                message=f"成功识别用户: {result[0]}",
                identified_user=result[0],
                confidence=result[1],
                feature_shape=str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )
        else:
            return RecognitionResponse(
                success=False,
                message="未能识别出匹配的用户",
                identified_user=None,
                confidence=None,
                feature_shape=str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )

    except binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"无效的base64编码: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}")


@app.post("/api/v1/vpr/recognize/file", response_model=RecognitionResponse)
async def identify_user_file(file: UploadFile = File(...), threshold: Optional[float] = None):
    """
    通过上传文件识别用户
    """
    try:
        # 检查文件格式 - 更宽松的检查
        if file.content_type and not file.content_type.startswith('audio/'):
            # 如果明确指定了非audio类型才拒绝
            if file.content_type not in ['application/octet-stream', 'application/x-download']:
                raise HTTPException(status_code=400, detail="只支持音频文件")

        # 检查文件扩展名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}，支持的格式: {', '.join(allowed_extensions)}")

        # 读取音频数据
        audio_bytes = await file.read()

        # 获取音频特征
        # embedding = predictor.predict(audio_data=audio_bytes)

        # 识别用户
        result = predictor.recognition(audio_data=audio_bytes, threshold=threshold)

        if result[0] is not None:
            return RecognitionResponse(
                success=True,
                message=f"成功识别用户: {result[0]}",
                identified_user=result[0],
                confidence=result[1],
                feature_shape='1'#str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )
        else:
            return RecognitionResponse(
                success=False,
                message="未能识别出匹配的用户",
                identified_user=None,
                confidence=None,
                feature_shape='1'#str(embedding.shape) if hasattr(embedding, 'shape') else str(type(embedding))
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理音频文件失败: {str(e)}")


@app.get("/api/v1/vpr/users", response_model=UserListResponse)
async def get_registered_users():
    """
    获取所有已注册的用户列表
    """
    try:
        users = predictor.get_users()
        unique_users = list(set(users))
        return UserListResponse(
            success=True,
            users=unique_users,
            count=len(unique_users)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取用户列表失败: {str(e)}")


@app.delete("/api/v1/vpr/users/{user_id}", response_model=DeleteUserResponse)
async def delete_user(user_id: str):
    """
    删除指定用户及其所有音频数据
    """
    try:
        success = predictor.remove_user(user_id)
        if success:
            return DeleteUserResponse(
                success=True,
                message=f"用户 {user_id} 已成功删除"
            )
        else:
            raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除用户失败: {str(e)}")


# @app.post("/api/v1/vpr/compare", response_model=CompareResponse)
# async def compare_voices(audio_data1: AudioData, audio_data2: AudioData, threshold: float = 0.6):
#     """
#     比较两个音频的相似度
#     """
#     try:
#         # 解码音频数据
#         audio_bytes1 = base64.b64decode(audio_data1.data)
#         audio_bytes2 = base64.b64decode(audio_data2.data)
#
#         # 计算相似度
#         similarity = predictor.contrast(audio_bytes1, audio_bytes2)
#         is_same_person = similarity >= threshold
#
#         return CompareResponse(
#             success=True,
#             message=f"音频相似度计算完成",
#             similarity=float(similarity),
#             is_same_person=is_same_person
#         )
#
#     except binascii.Error as e:
#         raise HTTPException(status_code=400, detail=f"无效的base64编码: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"音频对比失败: {str(e)}")
#
#
# @app.post("/api/v1/vpr/compare/files", response_model=CompareResponse)
# async def compare_voice_files(
#     file1: UploadFile = File(...),
#     file2: UploadFile = File(...),
#     threshold: float = 0.6
# ):
#     """
#     比较两个音频文件的相似度
#     """
#     try:
#         # 检查文件格式
#         if not file1.content_type or not file1.content_type.startswith('audio/'):
#             raise HTTPException(status_code=400, detail="文件1只支持音频文件")
#         if not file2.content_type or not file2.content_type.startswith('audio/'):
#             raise HTTPException(status_code=400, detail="文件2只支持音频文件")
#
#         # 读取音频数据
#         audio_bytes1 = await file1.read()
#         audio_bytes2 = await file2.read()
#
#         # 计算相似度
#         similarity = predictor.contrast(audio_bytes1, audio_bytes2)
#         is_same_person = similarity >= threshold
#
#         return CompareResponse(
#             success=True,
#             message=f"音频文件相似度计算完成",
#             similarity=float(similarity),
#             is_same_person=is_same_person
#         )
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"音频文件对比失败: {str(e)}")


@app.get("/")
async def root():
    """
    API根路径
    """
    return {
        "message": "语音识别API服务",
        "version": "1.0.0",
        "endpoints": {
            "register": "/api/v1/vpr/register - 注册用户音频(base64)",
            "register_file": "/api/v1/vpr/register/file - 注册用户音频(文件上传)",
            "recognize": "/api/v1/vpr/recognize - 识别用户音频(base64)",
            "recognize_file": "/api/v1/vpr/recognize/file - 识别用户音频(文件上传)",
            # "compare": "/api/v1/vpr/compare - 比较两个音频(base64)",
            # "compare_files": "/api/v1/vpr/compare/files - 比较两个音频文件",
            "users": "/api/v1/vpr/users - 获取已注册用户列表",
            "delete_user": "/api/v1/vpr/users/{user_id} - 删除用户",
            "docs": "/docs - API文档"
        }
    }