# -*- encoding: utf-8 -*-
"""
人脸识别系统 - 数据库访问层
=============================
基于原版 FaceSQL.py 重构，保持 MySQL（pymysql）与数据库表结构不变，
同时做了防御式改造：
  1. 数据库连接失败时不终止进程（原版会 raise SystemExit 导致服务崩溃），
     而是记录日志并返回空结果，保证接口不报错。
  2. 所有方法返回稳定的数据类型（list / (bool, str)），便于上层统一处理。
"""
import logging

import pymysql

from app import config

logger = logging.getLogger(__name__)


class FaceSQL:
    """学生人脸数据库操作封装（MySQL）。"""

    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        """建立 MySQL 连接，失败时记录日志（不抛出异常）。"""
        try:
            self.conn = pymysql.connect(**config.MYSQL_CONFIG)
            logger.info("人脸数据库连接成功: %s/%s", config.MYSQL_CONFIG["host"], config.MYSQL_CONFIG["db"])
        except Exception as exc:  # noqa: BLE001 - 连接失败不应中断服务
            logger.warning("人脸数据库连接失败: %s", exc)
            self.conn = None

    def close(self):
        """关闭数据库连接。"""
        if self.conn:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            finally:
                self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return exc_type is None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def all_face_data(self):
        """获取所有人脸数据：[(姓名, 面部特征图片二进制), ...]。连接失败返回 []。"""
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT `姓名`, `面部特征id` FROM `student`")
                return cursor.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询所有人脸数据失败: %s", exc)
            return []

    def all_student_data(self):
        """获取所有学生姓名列表。连接失败返回 []。"""
        if not self.conn:
            return []
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT `姓名` FROM `student`")
                return [row[0] for row in cursor.fetchall() if row[0] and str(row[0]).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询学生名单失败: %s", exc)
            return []

    def check_student_exists(self, student_id):
        """检查学号是否已存在。连接失败返回 False。"""
        if not self.conn:
            return False
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT `学号` FROM `student` WHERE `学号` = %s", (student_id,))
                return cursor.fetchone() is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning("检查学号失败: %s", exc)
            return False

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def save_face_data(self, student_id, name, class_name, image_data):
        """
        保存学生人脸数据（事务内完成学号检查 + 插入，避免重复）。
        返回 (success: bool, message: str)。
        """
        if not self.conn:
            return False, "数据库未连接，请检查 MySQL 服务是否开启"
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT `学号` FROM `student` WHERE `学号` = %s FOR UPDATE NOWAIT",
                    (student_id,),
                )
                if cursor.fetchone():
                    return False, f"学号 {student_id} 已存在，请勿重复提交"
                cursor.execute(
                    "INSERT INTO `student` (`学号`, `姓名`, `班级`, `面部特征id`) VALUES (%s, %s, %s, %s)",
                    (student_id, name, class_name, image_data),
                )
                self.conn.commit()
                return True, "学生信息保存成功"
        except pymysql.err.OperationalError as exc:
            if exc.args and exc.args[0] == 3572:  # NOWAIT 锁超时
                return False, f"学号 {student_id} 正在被其他操作修改，请稍后重试"
            return False, f"数据库错误: {exc}"
        except Exception as exc:  # noqa: BLE001
            try:
                self.conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            return False, f"保存失败: {exc}"
