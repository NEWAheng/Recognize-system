# -*- encoding: utf-8 -*-
"""
全局配置文件
=============
本模块集中管理项目的路径常量、模型权重目录、默认推理参数，
并对 ultralytics 源码包做导入支持（该源码包未以 pip 方式安装时使用）。
"""
from __future__ import annotations  # 兼容 Python 3.8（延后求值类型注解）

import importlib.util
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# 项目根目录：config.py 位于 app/ 目录，其上一级即为项目根目录
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# 目录配置
# ------------------------------------------------------------------
MODEL_DIR = Path(os.environ.get("YOLO_MODEL_DIR", str(BASE_DIR / "weights")))  # 模型权重目录
UPLOAD_DIR = BASE_DIR / "uploads"                                             # 视频上传临时目录
OUTPUT_FRAME_DIR = BASE_DIR / "output_frames"                                 # 帧保存输出目录
STATIC_DIR = BASE_DIR / "app" / "static"                                      # 前端静态资源目录

# 确保关键目录存在（上传/输出目录会在运行时动态创建）
for _dir in (UPLOAD_DIR, OUTPUT_FRAME_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# ultralytics 导入支持
# 优先级：已通过 pip 安装的 ultralytics > 本地源码目录（E:\ultralytics-main）。
# 说明：本地源码是较旧版本（8.0.x），若环境已 pip 安装新版，则不注入源码路径，
#       避免旧版源码覆盖新版导致兼容性问题。
# ------------------------------------------------------------------
if importlib.util.find_spec("ultralytics") is None:
    _ULTRALYTICS_SRC = Path(os.environ.get("ULTRALYTICS_SRC", r"E:\ultralytics-main"))
    if _ULTRALYTICS_SRC.exists() and str(_ULTRALYTICS_SRC) not in sys.path:
        sys.path.insert(0, str(_ULTRALYTICS_SRC))

# ------------------------------------------------------------------
# 模型列表：扫描权重目录下所有 .pt / .onnx 文件
# ------------------------------------------------------------------
def get_model_list() -> list[str]:
    """返回权重目录中可用的模型文件名列表。"""
    if not MODEL_DIR.exists():
        return []
    return sorted(
        p.name for p in MODEL_DIR.iterdir() if p.suffix.lower() in (".pt", ".onnx")
    )

MODEL_LIST = get_model_list()

# ------------------------------------------------------------------
# 默认推理参数（与原始 Streamlit 应用的默认值保持一致）
# ------------------------------------------------------------------
DEFAULT_CONF = 0.25  # 默认置信度阈值
DEFAULT_IOU = 0.45   # 默认 IoU 阈值

# ------------------------------------------------------------------
# 人脸识别系统配置（与原版 face_recognirion.py / FaceSQL.py 保持一致）
# ------------------------------------------------------------------
# dlib 模型目录：默认指向 ultralytics-main（原版所在位置），
# 如需项目自包含，可把两个 .dat 文件复制到本项目目录后修改该配置。
FACE_MODEL_DIR = Path(os.environ.get("FACE_MODEL_DIR", r"E:\ultralytics-main"))
FACE_SHAPE_PREDICTOR = FACE_MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = FACE_MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat"

# MySQL 配置（与原版 FaceSQL.py 一致，但密码通过环境变量注入，避免公开仓库泄露）
# 本地运行时需设置环境变量，例如 PowerShell：$env:FACE_MYSQL_PASSWORD="你的密码"
MYSQL_CONFIG = {
    "host": os.environ.get("FACE_MYSQL_HOST", "localhost"),
    "user": os.environ.get("FACE_MYSQL_USER", "root"),
    "password": os.environ.get("FACE_MYSQL_PASSWORD", ""),  # 必须通过环境变量注入
    "db": os.environ.get("FACE_MYSQL_DB", "真操蛋数据库"),
    "port": int(os.environ.get("FACE_MYSQL_PORT", "3306")),
    "charset": "utf8mb4",
}

# 本地人脸特征缓存文件（加速启动，避免每次重新提取）
FACE_FEATURE_CACHE = BASE_DIR / "face_features.pkl"

# 视频截取帧输出目录（位于静态目录内，可直接通过 URL 访问）
FACE_FRAME_OUTPUT_DIR = STATIC_DIR / "generated" / "face_frames"
FACE_FRAME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
