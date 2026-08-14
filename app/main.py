# -*- encoding: utf-8 -*-
"""
FastAPI 应用入口
================
组装路由、静态资源与中间件，构建双系统应用：
  1. 基于 YOLOv8 的学生课堂行为分析系统（图片/视频/摄像头检测）
  2. 人脸识别与视频截取系统（视频截取/人脸识别/添加新人脸）
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.routes import camera, face, image, video

app = FastAPI(
    title="课堂行为分析系统",
    description=(
        "由 Streamlit 多页应用重构为 FastAPI 版，包含：\n"
        "1. YOLOv8 识别系统：图片检测、视频检测、本地摄像头检测；\n"
        "2. 人脸识别系统：视频截取、人脸识别、添加新人脸。"
    ),
    version="2.0.0",
)

# ------------------------------------------------------------------
# 跨域中间件：便于前后端分离部署时本地联调
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 业务路由
# ------------------------------------------------------------------
app.include_router(image.router)
app.include_router(video.router)
app.include_router(camera.router)
app.include_router(face.router)


# ------------------------------------------------------------------
# 配置相关接口
# ------------------------------------------------------------------
@app.get("/api/models", tags=["配置"])
async def list_models():
    """返回可用模型列表与默认推理参数，供前端初始化配置面板。"""
    return {
        "code": 0,
        "models": config.MODEL_LIST,
        "default_conf": config.DEFAULT_CONF,
        "default_iou": config.DEFAULT_IOU,
    }


# ------------------------------------------------------------------
# 静态资源与首页
# ------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """返回前端首页。"""
    return FileResponse(config.STATIC_DIR / "index.html")
