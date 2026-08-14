# -*- encoding: utf-8 -*-
"""
本地摄像头检测接口
==================
GET /api/camera/stream —— 以 MJPEG 流形式实时返回摄像头推理结果
GET /api/camera/stop   —— 释放摄像头资源
"""
import cv2
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.services.camera import camera_manager
from app.services.detector import infer_video_frame, load_model

router = APIRouter(prefix="/api/camera", tags=["摄像头检测"])

BOUNDARY = "frame"


@router.get("/stream")
async def camera_stream(
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
    model: str = Query("", description="模型文件名（留空则自动选择）"),
):
    """开启摄像头并以 MJPEG 流形式实时返回推理结果。"""
    if not camera_manager.start():
        raise HTTPException(status_code=500, detail="无法打开本地摄像头，请检查摄像头是否可用")

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        _camera_generator(conf, iou, model),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


@router.get("/stop")
async def camera_stop():
    """停止摄像头推理，释放摄像头资源。"""
    camera_manager.stop()
    return {"code": 0, "message": "摄像头已释放"}


def _camera_generator(conf: float, iou: float, model_name: str):
    """
    摄像头帧生成器：读取一帧 -> 推理标注 -> 编码 JPEG -> 按 MJPEG 协议产出。
    客户端断开连接时自动停止循环并释放摄像头。
    """
    model = load_model(model_name)
    try:
        while True:
            frame = camera_manager.read()
            if frame is None:
                break

            anno_img = infer_video_frame(model, frame, conf, iou)
            ok_jpg, jpg = cv2.imencode(".jpg", anno_img)
            if ok_jpg:
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg.tobytes() + b"\r\n"
                )
    finally:
        # 客户端断开（或生成器被关闭）时释放摄像头资源
        camera_manager.stop()
