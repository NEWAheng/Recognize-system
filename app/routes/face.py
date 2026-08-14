# -*- encoding: utf-8 -*-
"""
人脸识别系统 - API 路由
========================
对应原始 Streamlit 页面 face_recognition_system.py 的三个功能：
  1. 视频截取（POST /api/face/video/extract）
  2. 人脸识别（POST /api/face/recognize）
  3. 添加新人脸（POST /api/face/add）
另提供学生名单与系统状态接口供前端展示。
"""
import base64
import io
import logging
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Pillow 缺失时服务仍可启动，识别接口会给出明确提示
    Image = ImageDraw = ImageFont = None

from app import config
from app.services.face_recognizer import get_face_recognizer
from app.services.face_sql import FaceSQL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/face", tags=["人脸识别系统"])

# 中文字体候选（Windows），用于在标注图上绘制姓名
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "simhei.ttf",
]


def _load_font(size: int = 20):
    """加载中文字体，全部失败则返回 None（回退到默认字体）。"""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return None


def _annotate_image(img_rgb, results):
    """
    在 RGB 图像上绘制人脸框与姓名标签。
    已知人员：绿色框；未知人员：红色框。
    返回标注后的 RGB ndarray。
    """
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(20)

    for name, similarity, face in results:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if name != "未知人员" else (255, 0, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        label = f"{name} ({similarity:.2f})"
        y_text = y - 20 if y - 20 > 20 else y + 20
        if font:
            draw.text((x, y_text), label, font=font, fill=color)
        else:
            draw.text((x, y_text), label, fill=color)

    return np.array(img_pil)


def _encode_jpeg_b64(img_rgb) -> str:
    """将 RGB ndarray 编码为 JPEG base64 字符串。"""
    img_pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bytes_to_rgb(data: bytes) -> np.ndarray:
    """将上传文件字节解码为 RGB ndarray，失败返回 None。"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------
# 1. 人脸识别
# ------------------------------------------------------------------
@router.post("/recognize")
async def recognize(
    files: list[UploadFile] = File(..., description="待识别的图片（可多张）"),
    confidence: float = Form(0.6, ge=0.0, le=1.0, description="识别置信度阈值"),
):
    """
    对上传的图片逐张进行人脸识别比对。
    返回每张图片的：原始图、标注图（base64）、识别统计、到场/缺席名单。
    """
    if Image is None:
        raise HTTPException(
            status_code=500,
            detail="Pillow 未安装，无法绘制识别结果。请执行：E:\\anaconda\\envs\\NEWlif\\python.exe -m pip install pillow",
        )

    try:
        recognizer = get_face_recognizer()
    except Exception as exc:  # noqa: BLE001 - 初始化失败（dlib 缺失/模型缺失）
        logger.warning("人脸识别器初始化失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"人脸识别器初始化失败: {exc}")

    total_students = []
    try:
        with FaceSQL() as db:
            total_students = db.all_student_data()
    except Exception:  # noqa: BLE001
        pass

    images = []
    for f in files:
        data = await f.read()
        img_rgb = _bytes_to_rgb(data)
        if img_rgb is None:
            images.append({
                "filename": f.filename or "未命名",
                "error": "图片无法解析",
            })
            continue

        results, img_rgb = recognizer.recognize_face(img_rgb, confidence_threshold=confidence)
        annotated = _annotate_image(img_rgb, results)

        known_count = sum(1 for r in results if r[0] != "未知人员")
        unknown_count = len(results) - known_count
        people_info = [
            {"name": r[0], "similarity": f"{r[1]:.4f}"}
            for r in results if r[0] != "未知人员"
        ]
        recognized_students = list({r[0] for r in results if r[0] != "未知人员"})
        absent_students = list(set(total_students) - set(recognized_students))

        images.append({
            "filename": f.filename or "未命名",
            "original_b64": _encode_jpeg_b64(img_rgb),
            "annotated_b64": _encode_jpeg_b64(annotated),
            "faces_count": len(results),
            "known_count": known_count,
            "unknown_count": unknown_count,
            "people_info": people_info,
            "recognized_students": recognized_students,
            "absent_students": absent_students,
        })

    return {
        "code": 0,
        "total_students_count": len(total_students),
        "images": images,
    }


# ------------------------------------------------------------------
# 2. 视频截取
# ------------------------------------------------------------------
@router.post("/video/extract")
async def extract_video_frames(
    file: UploadFile = File(..., description="待截取的视频文件（mp4/mov）"),
    start_time: float = Form(0.0, ge=0.0, description="开始时间（秒）"),
    end_time: float = Form(60.0, description="结束时间（秒）"),
    num_frames: int = Form(10, ge=1, le=50, description="截取帧数"),
):
    """
    在视频指定时间段内均匀截取若干帧图片，保存到静态目录并返回可访问的 URL。
    """
    suffix = "." + (file.filename.rsplit(".", 1)[-1] if "." in (file.filename or "") else "mp4")
    temp_path = config.UPLOAD_DIR / f"face_extract_{uuid.uuid4().hex}{suffix}"
    data = await file.read()
    temp_path.write_bytes(data)

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="视频无法解析，请上传正确的视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    if start_time >= end_time:
        raise HTTPException(status_code=400, detail="结束时间必须大于开始时间")

    start_frame = min(int(start_time * fps), total_frames - 1)
    end_frame = min(int(end_time * fps), total_frames - 1)
    if start_frame >= end_frame:
        raise HTTPException(status_code=400, detail="该时间段内没有可截取的帧")

    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    # 独立输出子目录，避免多人并发写冲突
    out_dir = config.FACE_FRAME_OUTPUT_DIR / uuid.uuid4().hex[:8]
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(temp_path))
    urls = []
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))
        ok, frame = cap.read()
        if not ok:
            continue
        filename = f"frame_{idx + 1:02d}.jpg"
        cv2.imwrite(str(out_dir / filename), frame)
        urls.append(f"/static/generated/face_frames/{out_dir.name}/{filename}")
    cap.release()

    # 清理临时上传文件
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass

    return {
        "code": 0,
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "total_frames": total_frames,
        "extracted": len(urls),
        "frame_urls": urls,
    }


# ------------------------------------------------------------------
# 3. 添加新人脸
# ------------------------------------------------------------------
@router.post("/add")
async def add_new_face(
    student_id: str = Form(..., description="学号（必填）"),
    name: str = Form(..., description="姓名（必填）"),
    class_name: str = Form("", description="班级（可选）"),
    file: UploadFile = File(..., description="人脸图片"),
):
    """添加新人脸到数据库，并立即更新识别缓存。"""
    student_id = student_id.strip()
    name = name.strip()
    if not student_id or not name:
        raise HTTPException(status_code=400, detail="学号和姓名不能为空")

    data = await file.read()
    img_rgb = _bytes_to_rgb(data)
    if img_rgb is None:
        raise HTTPException(status_code=400, detail="图片无法解析")

    try:
        recognizer = get_face_recognizer()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"人脸识别器初始化失败: {exc}")

    ok, message = recognizer.add_new_face(img_rgb, student_id, name, class_name)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    return {"code": 0, "message": message}


# ------------------------------------------------------------------
# 4. 学生名单 / 系统状态
# ------------------------------------------------------------------
@router.get("/students")
async def student_list():
    """返回数据库中的学生姓名列表。"""
    try:
        with FaceSQL() as db:
            names = db.all_student_data()
    except Exception:  # noqa: BLE001
        names = []
    return {"code": 0, "total": len(names), "students": names}


@router.get("/status")
async def face_status():
    """
    返回人脸识别系统可用性状态（dlib / 模型 / 数据库），
    供前端在加载时给出友好提示，避免静默失败。
    """
    from app.services import face_recognizer as fr_mod

    dlib_ok = fr_mod.dlib is not None
    model_ok = config.FACE_SHAPE_PREDICTOR.exists() and config.FACE_RECOGNITION_MODEL.exists()

    db_ok, total_students = False, 0
    try:
        with FaceSQL() as db:
            total_students = len(db.all_student_data())
            db_ok = db.conn is not None
    except Exception:  # noqa: BLE001
        pass

    message = "人脸识别系统就绪"
    if not dlib_ok:
        message = "dlib 未安装，人脸识别功能不可用"
    elif not model_ok:
        message = "dlib 模型文件缺失，请检查 FACE_MODEL_DIR 配置"
    elif not db_ok:
        message = "MySQL 未连接，识别将仅基于本地缓存特征"

    return {
        "code": 0,
        "dlib_ok": dlib_ok,
        "model_ok": model_ok,
        "db_ok": db_ok,
        "total_students": total_students,
        "message": message,
    }
