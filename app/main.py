from fastapi import FastAPI

from app.vpr import predictor

# 获取音频特征
# embedding = predictor.predict(audio_data='deps/vpr/dataset/a_1.wav')
# # 获取两个音频的相似度
# similarity = predictor.contrast(audio_data1='deps/vpr/dataset/a_1.wav', audio_data2='deps/vpr/dataset/a_2.wav')

# 注册用户音频
# predictor.register(user_name='夜雨飘零', audio_data='deps/vpr/dataset/b_1.wav')
# # 识别用户音频
# name, score = predictor.recognition(audio_data='deps/vpr/dataset/b_2.wav')
# # 获取所有用户
# users_name = predictor.get_users()
# # 删除用户音频
# predictor.remove_user(user_name='夜雨飘零')
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/user/{user_id}")
async def say_hello(user_id: str):
    return {"message": f"User: {user_id}", "users": predictor.get_users()}
