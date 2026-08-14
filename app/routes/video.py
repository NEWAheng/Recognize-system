# -*- encoding: utf-8 -*-
"""
视频检测接口
============
POST /api/detect/video/upload —— 上传视频文件，返回推理流地址
GET  /api/detect/video/stream  —— 以 MJPEG 流形式实时返回标注后的视频帧
"""
import uuid

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from app.services.detector import VideoFrameSaver, infer_video_frame, load_model

router = APIRouter(prefix="/api/detect", tags=["视频检测"])

# MJPEG 流的边界分隔符（multipart/x-mixed-replace 协议要求）
BOUNDARY = "frame"


@router.post("/video/upload")
async def upload_video(file: UploadFile = File(..., description="待检测的 mp4 视频")):
    """
    上传视频文件，保存为临时文件并返回推理流地址。

    Returns:
        code: 0 表示成功
        name: 临时文件名
        stream_url: 推理流地址，前端 <img> 标签直接引用即可播放
    """
    if file.content_type != "video/mp4" and not (file.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="仅支持 mp4 格式的视频")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传的视频为空")

    # 使用 uuid 生成唯一文件名，避免并发上传互相覆盖
    name = f"{uuid.uuid4().hex}.mp4"
    (config.UPLOAD_DIR / name).write_bytes(data)

    return {
        "code": 0,
        "message": "上传成功",
        "name": name,
        "stream_url": f"/api/detect/video/stream?name={name}",
    }


@router.get("/video/stream")
async def video_stream(
    name: str = Query(..., description="上传视频对应的文件名"),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="置信度阈值"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU 阈值"),
    model: str = Query("", description="模型文件名（留空则自动选择）"),
    save_frames: bool = Query(False, description="是否保存视频帧"),
    interval_minutes: int = Query(10, ge=1, description="帧保存间隔分钟数"),
    frames_per_minute: int = Query(10, ge=1, le=60, description="每分钟保存帧数"),
    output_folder: str = Query("output_frames", description="帧保存输出文件夹"),
    filename_format: str = Query("frame_{time}_{index}", description="帧文件名格式"),
):
    """以 MJPEG 流形式返回视频推理结果。"""
    video_path = config.UPLOAD_DIR / name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频不存在或已被清理")

    # 提前校验视频可读，避免推理过程中才发现错误
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="无法读取视频文件")
    cap.release()

    headers = {
        "Cache-Control": "no-cache",           # 禁止浏览器缓存，保证实时性
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",             # 告知反向代理不要缓冲（nginx 等）
    }
    return StreamingResponse(
        _mjpeg_generator(video_path, conf, iou, model, save_frames,
                         interval_minutes, frames_per_minute,
                         output_folder, filename_format),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )


def _mjpeg_generator(video_path, conf: float, iou: float, model_name: str,
                     save_frames: bool, interval_minutes: int,
                     frames_per_minute: int, output_folder: str,
                     filename_format: str):
    """
    视频推理帧生成器：逐帧读取 -> 推理标注 -> 编码 JPEG -> 按 MJPEG 协议产出。
    客户端断开连接时，finally 块会释放视频资源。
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saver = None
        if save_frames:
            saver = VideoFrameSaver(
                fps=fps, total_frames=total_frames,
                interval_minutes=interval_minutes,
                frames_per_minute=frames_per_minute,
                output_folder=output_folder,
                filename_format=filename_format,
            )

        model = load_model(model_name)
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            anno_img = infer_video_frame(model, frame, conf, iou)
            if saver:
                saver.step(anno_img)

            ok_jpg, jpg = cv2.imencode(".jpg", anno_img)
            if ok_jpg:
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg.tobytes() + b"\r\n"
                )
    finally:
        cap.release()
