# -*- encoding: utf-8 -*-
"""
检测服务模块
============
负责 YOLO 模型的加载（进程内缓存）与图片/视频帧推理，
以及视频帧定时保存功能（复刻原 Streamlit 应用的帧保存逻辑）。
"""
from __future__ import annotations  # 延迟求值类型注解

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import cv2

from app import config


# ------------------------------------------------------------------
# 模型加载
# ------------------------------------------------------------------
@lru_cache(maxsize=2)
def load_model(model_name: str = "") -> YOLO:
    """
    加载 YOLO 模型（结果被缓存，同一模型只加载一次）。

    选择优先级：
        1. 显式指定的 model_name
        2. 权重目录中的 .pt 权重（PyTorch 推理）
        3. 权重目录中的 .onnx 权重（ONNX Runtime 推理）

    Raises:
        FileNotFoundError: 权重目录中没有任何可用模型文件
    """
    candidates = [model_name] if model_name else config.MODEL_LIST
    if not candidates:
        raise FileNotFoundError("权重目录中未找到任何模型文件(.pt/.onnx)")

    pt_files = [f for f in candidates if f.lower().endswith(".pt")]
    onnx_files = [f for f in candidates if f.lower().endswith(".onnx")]

    selected = None
    if model_name:
        path = config.MODEL_DIR / model_name
        if path.exists():
            selected = str(path)
    if selected is None and pt_files:
        selected = str(config.MODEL_DIR / pt_files[0])
    if selected is None and onnx_files:
        selected = str(config.MODEL_DIR / onnx_files[0])
    if selected is None:
        raise FileNotFoundError(f"未找到模型文件: {model_name or candidates}")

    # 延迟导入：未安装 torch/ultralytics 时服务仍可启动，仅在调用检测时给出明确错误
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics 未安装，无法使用 YOLO 检测功能。"
            "请先安装依赖：E:\\anaconda\\envs\\NEWlif\\python.exe -m pip install torch torchvision ultralytics"
        ) from exc

    return YOLO(selected)


# ------------------------------------------------------------------
# 图片推理
# ------------------------------------------------------------------
def infer_image(model: YOLO, image, conf: float, iou: float):
    """
    对单张图片进行推理。

    Args:
        model: 已加载的 YOLO 模型
        image: 图像数组（BGR，numpy.ndarray）
        conf: 置信度阈值
        iou: IoU 阈值

    Returns:
        anno_img: 绘制了检测框的标注图像（BGR）
        label_counts: 各类别检出数量，形如 {"student": 3, ...}
        rows: 检测明细列表，每项为 [序号, 类别, 置信度, {x1,y1,x2,y2}]
    """
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    res = results[0]
    anno_img = res.plot()
    labels = res.names
    boxes = res.boxes

    label_counts: dict = {}
    rows: list = []
    if boxes is not None:
        for index, box in enumerate(boxes):
            cls_index = int(box.cls.cpu().numpy()[0])
            label_name = labels[cls_index]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy.cpu().tolist()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            rows.append([
                index + 1,
                label_name,
                f"{confidence:.2f}",
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            ])
            label_counts[label_name] = label_counts.get(label_name, 0) + 1

    return anno_img, label_counts, rows


# ------------------------------------------------------------------
# 视频/摄像头单帧推理
# ------------------------------------------------------------------
def infer_video_frame(model: YOLO, image, conf: float, iou: float):
    """对视频单帧进行推理，返回绘制了检测框的标注图像（BGR）。"""
    res = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    return res[0].plot()


# ------------------------------------------------------------------
# 视频帧保存（复刻原 Streamlit 应用的定时帧保存功能）
# ------------------------------------------------------------------
class VideoFrameSaver:
    """
    按原应用的规则在视频推理过程中定时保存帧图片。

    规则说明：
        在视频每个“间隔周期”的前 minute_frames 帧内，以及视频最后
        minute_frames 帧内，每隔 frame_interval 帧保存一张标注图。
    """

    def __init__(self, fps: float, total_frames: int,
                 interval_minutes: int, frames_per_minute: int,
                 output_folder: str, filename_format: str):
        self.fps = fps or 25.0
        self.total_frames = total_frames
        self.frames_per_minute = max(1, int(frames_per_minute))
        self.interval_frames = max(1, int(interval_minutes * 60 * self.fps))
        self.minute_frames = int(60 * self.fps)
        self.frame_interval = max(1, self.minute_frames // self.frames_per_minute)

        # 输出目录：相对路径统一放到项目 output_frames 下，避免路径穿越
        if output_folder and Path(output_folder).is_absolute():
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = config.OUTPUT_FRAME_DIR / (output_folder or "output_frames")
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.filename_format = filename_format
        self.frame_count = 0      # 已保存帧数
        self.interval_count = 0   # 已跨越的间隔周期数
        self.position = 0         # 当前帧位置

    def step(self, anno_img) -> bool:
        """
        处理当前帧：判断是否保存，并推进帧位置。

        Returns:
            是否保存了当前帧
        """
        saved = False
        in_head = (self.position % self.interval_frames) < self.minute_frames
        in_tail = self.position >= (self.total_frames - self.minute_frames)
        if in_head or in_tail:
            if (self.position % self.interval_frames) % self.frame_interval == 0:
                self._save(anno_img)
                saved = True
        if self.position > 0 and self.position % self.interval_frames == 0:
            self.interval_count += 1
        self.position += 1
        return saved

    def _save(self, anno_img) -> None:
        """按文件名格式生成文件名并写入磁盘。"""
        time_str = str(timedelta(seconds=self.position / self.fps)).replace(":", "-").split(".")[0]
        filename = self.filename_format.format(
            time=time_str,
            index=self.frame_count % self.frames_per_minute,
            total_index=self.frame_count,
            interval=self.interval_count,
        ) + ".jpg"
        cv2.imwrite(str(self.output_folder / filename), anno_img)
        self.frame_count += 1
