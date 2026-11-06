"""轻量启动入口：python review.py [参数]。"""
from backend.server import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f"启动失败：{error}\n依赖缺失时请在当前Python环境安装requirements.txt。") from error
