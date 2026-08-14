# -*- encoding: utf-8 -*-
"""
人脸识别系统 - 识别服务
========================
基于原版 face_recognirion.py 的 FaceRecognizer 类重构：
  1. 移除 streamlit 依赖（st.error/st.warning 等改为日志输出）。
  2. sklearn 的 cosine_similarity 用 numpy 等价实现替换。
  3. 数据库连接采用防御式处理：失败时仅使用本地特征缓存，接口不崩溃。
  4. dlib 采用延迟可用判断：未安装 dlib 时给出明确错误信息而非让服务崩溃。
"""
import logging
import pickle
import threading
from pathlib import Path

import numpy as np

try:
    import dlib
except ImportError:  # pragma: no cover - dlib 未安装时不影响服务启动
    dlib = None

from app import config
from app.services.face_sql import FaceSQL

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a, vec_b):
    """计算两个一维向量的余弦相似度（sklearn 等价实现，避免额外依赖）。"""
    a = np.asarray(vec_a).flatten()
    b = np.asarray(vec_b).flatten()
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


class FaceRecognizer:
    """dlib 人脸检测 + 特征提取 + 相似度识别。"""

    def __init__(self):
        if dlib is None:
            raise RuntimeError(
                "dlib 未安装，无法使用人脸识别功能。"
                "请先执行：E:\\anaconda\\envs\\NEWlif\\python.exe -m pip install dlib"
            )

        # 校验 dlib 模型文件
        if not config.FACE_SHAPE_PREDICTOR.exists():
            raise FileNotFoundError(f"人脸关键点模型不存在: {config.FACE_SHAPE_PREDICTOR}")
        if not config.FACE_RECOGNITION_MODEL.exists():
            raise FileNotFoundError(f"人脸识别模型不存在: {config.FACE_RECOGNITION_MODEL}")

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(config.FACE_SHAPE_PREDICTOR))
        self.face_rec_model = dlib.face_recognition_model_v1(str(config.FACE_RECOGNITION_MODEL))

        # 已知人脸：姓名 -> 128 维特征向量
        self.known_faces = {}

        # 从数据库 + 本地缓存加载已知人脸
        self.load_known_faces_from_db()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def load_known_faces_from_db(self):
        """从数据库加载已知人脸特征（含本地缓存加速）。"""
        # 1. 优先加载本地特征缓存
        cache_file = Path(config.FACE_FEATURE_CACHE)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.known_faces.update(pickle.load(f))
                logger.info("已从缓存加载 %d 个人脸特征", len(self.known_faces))
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取人脸特征缓存失败: %s", exc)

        # 2. 从数据库加载缺失的特征
        try:
            with FaceSQL() as db:
                students = db.all_face_data()
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载数据库人脸数据失败: %s", exc)
            return

        for name, image_data in students:
            name = str(name).strip()
            if not name or name in self.known_faces:
                continue
            img = self.load_image_from_bytes(image_data)
            if img is None:
                continue
            success, face_descriptor = self.extract_face_encoding(img)
            if success:
                self.known_faces[name] = face_descriptor
                self.save_features()
            else:
                logger.warning("无法从学生 %s 的图片中提取人脸特征", name)

        logger.info("人脸识别器就绪，已知人脸: %d", len(self.known_faces))

    def save_features(self):
        """保存人脸特征到本地缓存文件。"""
        try:
            with open(config.FACE_FEATURE_CACHE, "wb") as f:
                pickle.dump(self.known_faces, f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存人脸特征缓存失败: %s", exc)

    def reload_from_database(self):
        """从数据库重新加载所有人脸数据。"""
        logger.info("重新加载人脸数据...")
        self.load_known_faces_from_db()

    # ------------------------------------------------------------------
    # 图像处理
    # ------------------------------------------------------------------
    @staticmethod
    def load_image_from_bytes(image_data):
        """从字节数据加载图片，转换为 dlib 所需的 RGB 数组。失败返回 None。"""
        try:
            import cv2

            img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as exc:  # noqa: BLE001
            logger.warning("图片加载错误: %s", exc)
            return None

    @staticmethod
    def to_rgb_array(image):
        """将 PIL Image / numpy 数组统一转为 RGB ndarray。"""
        if hasattr(image, "convert"):  # PIL.Image
            return np.array(image.convert("RGB"))
        arr = np.asarray(image)
        if arr.ndim == 2:  # 灰度图
            return np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:  # RGBA -> RGB
            return arr[:, :, :3]
        return arr.copy()

    def extract_face_encoding(self, img):
        """
        从图像中提取最大人脸的 128 维特征。
        返回 (success: bool, face_descriptor or None)。
        """
        try:
            dets = self.detector(img, 1)
            if len(dets) == 0:
                return False, None
            face = max(dets, key=lambda rect: rect.width() * rect.height())
            shape = self.predictor(img, face)
            descriptor = self.face_rec_model.compute_face_descriptor(img, shape)
            return True, np.array(descriptor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("特征提取错误: %s", exc)
            return False, None

    def recognize_face(self, image, confidence_threshold=0.6):
        """
        识别图像中的人脸。
        返回 (results, img_rgb)：
          results = [(姓名, 相似度, dlib_rectangle), ...]
          img_rgb = RGB ndarray（用于后续标注绘制）
        """
        img_rgb = self.to_rgb_array(image)

        faces = self.detector(img_rgb, 1)
        if len(faces) == 0:
            return [], img_rgb

        results = []
        for face in faces:
            shape = self.predictor(img_rgb, face)
            descriptor = np.array(
                self.face_rec_model.compute_face_descriptor(img_rgb, shape)
            )

            best_match, highest_similarity = None, -1.0
            for name, known_face in self.known_faces.items():
                similarity = cosine_similarity(descriptor, known_face)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = name

            if best_match is not None and highest_similarity > confidence_threshold:
                results.append((best_match, highest_similarity, face))
            else:
                results.append(("未知人员", highest_similarity, face))

        return results, img_rgb

    # ------------------------------------------------------------------
    # 添加新人脸
    # ------------------------------------------------------------------
    def add_new_face(self, image, student_id, name, class_name=None):
        """
        将新人脸保存到数据库并更新本地缓存。
        返回 (success: bool, message: str)。
        """
        try:
            success, face_encoding = self.extract_face_encoding(self.to_rgb_array(image))
            if not success:
                return False, "未能从图片中检测到人脸，请上传清晰的单人正脸照片"

            import cv2

            img_np = np.asarray(image)
            if hasattr(image, "convert"):  # PIL.Image -> BGR ndarray
                img_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_np.copy()

            _, img_bytes = cv2.imencode(".jpg", img_bgr)
            image_binary = img_bytes.tobytes()

            with FaceSQL() as db:
                ok, message = db.save_face_data(
                    student_id=student_id,
                    name=name,
                    class_name=class_name,
                    image_data=image_binary,
                )
                if ok:
                    # 同时以学号和姓名为键更新本地缓存，保证识别立即生效
                    self.known_faces[name] = face_encoding
                    self.known_faces[student_id] = face_encoding
                    self.save_features()
                return ok, message
        except Exception as exc:  # noqa: BLE001
            logger.warning("添加新人脸出错: %s", exc)
            return False, f"添加人脸时出错: {exc}"


# ------------------------------------------------------------------
# 单例访问（线程安全、懒加载）
# ------------------------------------------------------------------
_recognizer = None
_recognizer_lock = threading.Lock()


def get_face_recognizer():
    """获取全局唯一的人脸识别器实例（首次调用时初始化）。"""
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = FaceRecognizer()
    return _recognizer
