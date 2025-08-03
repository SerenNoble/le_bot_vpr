from deps.vpr.mvector.predict import MVectorPredictor
from fastapi import FastAPI

predictor = MVectorPredictor(configs='app/vpr/configs/eres2net.yml',
                             model_path='models/CAMPPlus_Fbank/best_model/')
# 获取音频特征
embedding = predictor.predict(audio_data='dataset/a_1.wav')
# 获取两个音频的相似度
similarity = predictor.contrast(audio_data1='dataset/a_1.wav', audio_data2='dataset/a_2.wav')

# 注册用户音频
predictor.register(user_name='夜雨飘零', audio_data='dataset/test.wav')
# 识别用户音频
name, score = predictor.recognition(audio_data='dataset/test1.wav')
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
