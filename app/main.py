import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deps', 'vpr'))

from deps.vpr.mvector.predict import MVectorPredictor
from fastapi import FastAPI

predictor = MVectorPredictor(configs='app/vpr/configs/eres2net.yml',
                             model_path='app/vpr/models/ERes2Net_Fbank/',
                             use_gpu=False)
# 获取音频特征
embedding = predictor.predict(audio_data='deps/vpr/dataset/a_1.wav')
# 获取两个音频的相似度
similarity = predictor.contrast(audio_data1='deps/vpr/dataset/a_1.wav', audio_data2='deps/vpr/dataset/a_2.wav')

# 注册用户音频
predictor.register(user_name='夜雨飘零', audio_data='deps/vpr/dataset/b_1.wav')
# 识别用户音频
name, score = predictor.recognition(audio_data='deps/vpr/dataset/b_2.wav')
# 获取所有用户
users_name = predictor.get_users()
# 删除用户音频
predictor.remove_user(user_name='夜雨飘零')
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
