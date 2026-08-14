# -*- encoding: utf-8 -*-
"""
项目启动入口
============
使用方式：
    1. 切换到包含依赖的 Python 环境（本项目使用 SJSJ 环境）
    2. 在项目根目录执行：python run.py
    3. 浏览器访问 http://127.0.0.1:8001
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
    )
