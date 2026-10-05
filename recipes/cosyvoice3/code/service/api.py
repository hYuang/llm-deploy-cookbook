# CosyVoice3 服务 API 组装（由 app.py 拆分而来）

from fastapi import FastAPI

from .handlers import create_router


def create_app(model, default_prompt_wav: dict, upload_dir: str) -> FastAPI:
    app = FastAPI()
    app.include_router(create_router(model, default_prompt_wav, upload_dir))
    return app
