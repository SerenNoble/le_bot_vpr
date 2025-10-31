# 语音识别API系统

基于FastAPI和MVector的语音识别和说话人识别系统。

## 功能特性

- 🎤 **说话人注册**: 支持base64和文件上传两种方式
- 🔍 **说话人识别**: 高精度识别已注册用户
- 👥 **用户管理**: 查看、删除已注册用户
- 🔄 **语音对比**: 比较两个音频的相似度
- 📊 **RESTful API**: 标准的REST接口设计

## 快速开始

### 1. 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
# 创建虚拟环境（可选）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装主要依赖
pip install -r requirements.txt

# 安装VPR依赖
pip install -r deps/vpr/requirements.txt
```

### 3. 启动服务

```bash
# 方式1: 使用启动脚本
python run_server.py

# 方式2: 直接使用uvicorn
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问服务

- **API服务**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API接口说明

### 注册接口

#### 1. Base64方式注册
```http
POST /api/v1/vpr/register?user_id=张三
Content-Type: application/json

{
  "data": "base64编码的wav音频数据",
  "format": "wav",
  "codec": "pcm"
}
```

#### 2. 文件上传方式注册
```http
POST /api/v1/vpr/register/file?user_id=张三
Content-Type: multipart/form-data

file: [音频文件]
```

### 识别接口

#### 1. Base64方式识别
```http
POST /api/v1/vpr/recognize?threshold=0.6
Content-Type: application/json

{
  "data": "base64编码的wav音频数据",
  "format": "wav",
  "codec": "pcm"
}
```

#### 2. 文件上传方式识别
```http
POST /api/v1/vpr/recognize/file?threshold=0.6
Content-Type: multipart/form-data

file: [音频文件]
```

### 用户管理接口

#### 获取用户列表
```http
GET /api/v1/vpr/users
```

#### 删除用户
```http
DELETE /api/v1/vpr/users/张三
```

### 语音对比接口

#### Base64方式对比
```http
POST /api/v1/vpr/compare?threshold=0.6
Content-Type: application/json

{
  "audio_data1": {
    "data": "音频1的base64数据",
    "format": "wav",
    "codec": "pcm"
  },
  "audio_data2": {
    "data": "音频2的base64数据",
    "format": "wav",
    "codec": "pcm"
  }
}
```

#### 文件上传方式对比
```http
POST /api/v1/vpr/compare/files?threshold=0.6
Content-Type: multipart/form-data

file1: [音频文件1]
file2: [音频文件2]
```

## 测试

使用提供的测试脚本进行API测试：

```bash
# 运行测试脚本
python test_api.py
```

## 项目结构

```
lebot/
├── app/
│   ├── main.py              # FastAPI应用主文件
│   └── vpr/
│       ├── __init__.py      # 语音识别模块初始化
│       ├── configs/         # 模型配置文件
│       └── models/          # 预训练模型文件
├── audio_db/                # 音频数据库(自动创建)
├── deps/vpr/                # 语音识别依赖库
├── requirements.txt         # 项目依赖
├── run_server.py           # 服务启动脚本
├── test_api.py             # API测试脚本
└── README.md               # 项目说明文档
```

## 注意事项

1. **音频格式**: 推荐使用WAV格式，采样率16kHz
2. **音频时长**: 建议3-10秒，过短或过长都可能影响识别效果
3. **阈值设置**: 默认识别阈值为0.6，可根据实际情况调整
4. **存储空间**: 音频数据会保存在`audio_db`目录下

## 常见问题

### Q: 启动时提示模型文件不存在
A: 请检查`app/vpr/models/ERes2Net_Fbank/best_model/`目录下是否有模型文件

### Q: 识别准确率不高
A: 尝试以下方法：
- 使用清晰的录音环境
- 确保音频时长合适（3-10秒）
- 调整识别阈值
- 为每个用户注册多个音频样本

### Q: 支持哪些音频格式
A: 目前主要支持WAV格式，其他格式可能需要转换