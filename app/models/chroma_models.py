"""
ChromaDB数据模型定义 - 简化版本
专为ChromaDB向量存储设计的数据结构
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RecognitionResult(BaseModel):
    """识别结果模型"""
    success: bool
    user_id: Optional[str] = None
    voice_id: Optional[str] = None  # ChromaDB中的向量ID
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    is_user: Optional[bool] = None  # 是否是用户本人
    confidence: Optional[float] = None
    similarity: Optional[float] = None  # 相似度分数
    distance: Optional[float] = None   # 向量距离
    match_details: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None


class PersonInfo(BaseModel):
    """人员信息模型"""
    person_id: str
    person_name: str
    audio_count: int
    created_at: str


class UserStats(BaseModel):
    """用户统计信息"""
    user_id: str
    user_name: Optional[str] = None
    user_audio_count: int = 0
    total_persons: int = 0
    total_audio_features: int = 0
    persons_detail: List[PersonInfo] = []
    last_updated: datetime


class UserInfo(BaseModel):
    """用户信息模型"""
    user_id: str
    user_name: Optional[str] = None
    user_audio_count: int = 0
    total_persons: int = 0
    created_at: str


class UserListResponse(BaseModel):
    """用户列表响应模型"""
    success: bool
    users: List[UserInfo]
    count: int


class RegisterResponse(BaseModel):
    """注册响应模型"""
    success: bool
    message: str
    user_id: str
    person_name: str
    voice_id: Optional[str] = None
    registration_time: str


class RecognitionResponse(BaseModel):
    """识别响应模型"""
    success: bool
    message: str
    user_id: Optional[str] = None
    voice_id: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    is_user: Optional[bool] = None  # 是否是用户本人
    confidence: Optional[float] = None
    similarity: Optional[float] = None
    processing_time_ms: Optional[float] = None
    match_details: Optional[Dict[str, Any]] = None


class StorageInfo(BaseModel):
    """存储信息模型"""
    storage_type: str  # "single" 或 "multi_collection"
    total_users: int = 0
    base_directory: str
    total_vectors: Optional[int] = None
    collection_name: Optional[str] = None  # 单集合模式使用
    hnsw_space: str
    collections_per_user: Optional[int] = None  # 多集合模式使用


class StatsResponse(BaseModel):
    """统计信息响应模型"""
    success: bool = True
    stats: Dict[str, Any]


class GlobalStats(BaseModel):
    """全局统计信息"""
    total_users: int
    total_persons: int
    total_audio_features: int
    users_count: int
    last_updated: datetime