from fastapi import FastAPI
from pydantic import BaseModel
import base64
import binascii

from app.vpr import predictor

app = FastAPI()


class AudioData(BaseModel):
    format: str
    codec: str
    data: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/user/{user_id}")
async def say_hello(user_id: str):
    return {"message": f"User: {user_id}", "users": predictor.get_users()}


@app.post("/user/{user_id}")
async def register_user_audio(user_id: str, audio_data: AudioData):
    """
    上传base64编码的音频数据，获取音频特征并注册用户
    JSON格式: {"format": "wav", "codec": "pcm", "data": "base64 binary string"}
    """
    # 检查音频格式
    if audio_data.format.lower() != "wav":
        return {"error": "Only WAV format is supported"}

    try:
        # 解码base64音频数据为字节
        audio_bytes = base64.b64decode(audio_data.data)

        # 使用predictor.predict获取音频特征，直接传递字节数据
        embedding = predictor.predict(audio_data=audio_bytes)
        print(f"音频特征信息: {embedding}")

        # 使用predictor.register注册用户音频，直接传递字节数据
        predictor.register(user_name=user_id, audio_data=audio_bytes)

        return {
            "message": f"User {user_id} registered successfully",
            "audio_features_shape": embedding.shape if hasattr(embedding, 'shape') else str(type(embedding)),
            "user_id": user_id,
            "format": audio_data.format,
            "codec": audio_data.codec
        }

    except binascii.Error as e:
        return {"error": f"Invalid base64 encoding: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to process audio: {str(e)}"}
