# -*- encoding: utf-8 -*-
"""
图片检测接口
============
POST /api/detect/image —— 上传图片，返回标注结果图 + 检测明细 + 类别统计
"""
import base64

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.detector import infer_image, load_model

router = APIRouter(prefix="/api/detect", tags=["图片检测"])


@router.post("/image")
async def detect_image(
    file: UploadFile = File(..., description="待检测的图片文件"),
    conf: float = Form(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Form(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
    model: str = Form("", description="模型文件名（留空则自动选择）"),
):
    """
    上传一张图片进行 YOLO 推理。

    Returns:
        code: 0 表示成功
        image_base64: 标注后图像的 base64 编码（JPEG 格式）
        rows: 检测明细列表 [[序号, 类别, 置信度, {x1,y1,x2,y2}], ...]
        label_counts: 各类别检出数量
        total: 总检出目标数
    """
    # 校验文件类型
    if file.content_type not in ("image/png", "image/jpeg"):
        raise HTTPException(status_code=400, detail="仅支持 png/jpeg/jpg 格式的图片")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的图片为空")

    # 解码为 BGR 图像数组
    img_array = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img_array is None:
        raise HTTPException(status_code=400, detail="无法解析图片文件")

    # 推理
    model_obj = load_model(model)
    anno_img, label_counts, rows = infer_image(model_obj, img_array, conf, iou)

    # 标注图转 base64 返回给前端
    _, jpg = cv2.imencode(".jpg", anno_img)
    image_b64 = base64.b64encode(jpg.tobytes()).decode("ascii")

    return {
        "code": 0,
        "message": "检测成功",
        "image_base64": image_b64,
        "rows": rows,
        "label_counts": label_counts,
        "total": len(rows),
    }
