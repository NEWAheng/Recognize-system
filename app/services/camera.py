# -*- encoding: utf-8 -*-
"""
本地摄像头管理模块
==================
使用单例模式 + 线程锁封装本地摄像头的开启/读取/释放，
保证多个请求并发访问摄像头时线程安全。
"""
import threading

import cv2


class CameraManager:
    """封装本地摄像头的生命周期管理。"""

    def __init__(self, source: int = 0):
        self.source = source          # 摄像头编号，0 表示默认摄像头
        self._cap = None              # cv2.VideoCapture 实例
        self._lock = threading.Lock() # 互斥锁，保证多线程安全

    def start(self) -> bool:
        """打开摄像头，返回是否成功。"""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
            return self._cap.isOpened()

    def read(self):
        """读取一帧图像；失败时返回 None。"""
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            ok, frame = self._cap.read()
            return frame if ok else None

    def stop(self) -> None:
        """释放摄像头资源。"""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None


# 全局单例：所有摄像头相关路由共享同一个实例
camera_manager = CameraManager()
