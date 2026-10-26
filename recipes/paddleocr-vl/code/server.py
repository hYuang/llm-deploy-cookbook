# PaddleOCR-VL OpenAI 兼容包装服务（vLLM 后端）
#
# 说明：vLLM 自带的 /v1/chat/completions 已经是 OpenAI 兼容接口，常规调用**不需要本脚本**。
# 本脚本只在需要下面这些能力时用：
#   - multipart 直接上传图片，不用客户端自己转 base64
#   - /v1/chat/completions 自动补任务前缀（OCR: / Table Recognition: ...）
#   - 可选：PP-DocLayoutV2 整页版面解析（图片/PDF -> markdown）
#
# 部署：先按 ../README.md 启动 vLLM（默认 http://127.0.0.1:8000/v1），再启动本服务
#   pip install -r ../env/requirements.txt
#   cd <server.py 所在目录>
#   python server.py                     # 不带参数：读同目录 config.yaml
#   python server.py --config xxx.yaml   # 或指定配置文件
#
# 端点：
#   GET  /v1/models                OpenAI 模型列表
#   POST /v1/chat/completions      OpenAI 兼容透传（自动补任务前缀）
#   POST /ocr                      multipart：file + task，返回纯文本
#   POST /parse                    可选：PP-DocLayoutV2 整页解析，返回 markdown
#   GET  /health                   健康检查

import argparse
import base64
import os
import shutil
import sys
import tempfile
import uuid
from typing import Any, Dict, List

import requests
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型只认这几个任务前缀（官方 recipe），且必须是最后一段 text
TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}
_TASK_PROMPTS = frozenset(TASKS.values())

_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def load_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        sys.exit(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not (cfg.get("backend") or {}).get("base_url"):
        sys.exit("config missing backend.base_url")
    return cfg


def data_url(path: str) -> str:
    # 本地图片转 data URL，vLLM 的 image_url 字段直接吃
    mime = _MIME.get(os.path.splitext(path)[1].lower(), "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def task_prompt(name: str) -> str:
    # 未知任务名回落到 OCR，避免模型收到自然语言指令导致效果变差
    return TASKS.get((name or "").lower(), TASKS["ocr"])


async def save_upload(file: UploadFile) -> str:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    ext = os.path.splitext(file.filename or "upload.png")[1].lower() or ".png"
    path = os.path.join(tempfile.gettempdir(), f"paddleocr_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(data)
    return path


class Backend:
    def __init__(self, cfg: Dict[str, Any]):
        self.base = (cfg.get("base_url") or "").rstrip("/")
        self.model = cfg.get("model") or "PaddleOCR-VL"
        self.api_key = cfg.get("api_key") or "EMPTY"
        self.timeout = cfg.get("timeout", 3600)

    def post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = requests.post(
                f"{self.base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"vLLM unreachable at {self.base}: {e}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"vLLM error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def text(self, content: List[Dict[str, Any]], **extra) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": content}],
        }
        payload.update(extra)
        return self.post(payload)["choices"][0]["message"]["content"]


def create_app(cfg: Dict[str, Any]) -> FastAPI:
    backend = Backend(cfg.get("backend") or {})
    default_task = (cfg.get("defaults") or {}).get("task", "ocr")
    pipe_cfg = cfg.get("pipeline") or {}
    app = FastAPI(title="PaddleOCR-VL")

    # PP-DocLayoutV2 pipeline 懒加载：没装 paddleocr 也不影响纯 vLLM 路径
    _pipeline: List[Any] = []

    def get_pipeline():
        if not pipe_cfg.get("enabled"):
            raise HTTPException(
                status_code=501,
                detail="pipeline disabled: set pipeline.enabled=true and install paddleocr[doc-parser]",
            )
        if not pipe_cfg.get("layout_detection_model_dir"):
            raise HTTPException(status_code=501, detail="config missing pipeline.layout_detection_model_dir")
        if not _pipeline:
            try:
                from paddleocr import PaddleOCRVL
            except ImportError as e:
                raise HTTPException(status_code=501, detail=f"paddleocr not installed: {e}")
            _pipeline.append(
                PaddleOCRVL(
                    vl_rec_backend="vllm-server",
                    vl_rec_server_url=backend.base,
                    layout_detection_model_name="PP-DocLayoutV2",
                    layout_detection_model_dir=pipe_cfg["layout_detection_model_dir"],
                )
            )
        return _pipeline[0]

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": backend.base, "model": backend.model}

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": backend.model, "object": "model", "created": 0, "owned_by": "paddleocr-vl"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages required")

        # 最后一条消息是 content 数组时，检查是否已带任务前缀；没有就补默认任务
        content = messages[-1].get("content")
        if isinstance(content, list):
            texts = [p.get("text") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            if not any(t in _TASK_PROMPTS for t in texts):
                content.append({"type": "text", "text": task_prompt(default_task)})

        # 统一模型名，避免与 vllm serve --served-model-name 不一致
        body["model"] = backend.model
        body.setdefault("temperature", 0.0)
        return JSONResponse(content=backend.post(body))

    @app.post("/ocr")
    async def ocr(file: UploadFile = File(...), task: str = Form(default_task)):
        path = await save_upload(file)
        try:
            text = backend.text([
                {"type": "image_url", "image_url": {"url": data_url(path)}},
                {"type": "text", "text": task_prompt(task)},
            ])
        finally:
            if os.path.isfile(path):
                os.unlink(path)
        return PlainTextResponse(text)

    @app.post("/parse")
    async def parse(file: UploadFile = File(...)):
        path = await save_upload(file)
        outdir = tempfile.mkdtemp(prefix="paddleocr_")
        try:
            pages = []
            for i, res in enumerate(get_pipeline().predict(path)):
                md_path = os.path.join(outdir, f"page_{i}.md")
                res.save_to_markdown(save_path=md_path)
                markdown = ""
                if os.path.isfile(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        markdown = f.read()
                pages.append({"index": i, "markdown": markdown})
            return {"pages": pages, "markdown": "\n\n".join(p["markdown"] for p in pages)}
        finally:
            shutil.rmtree(outdir, ignore_errors=True)
            if os.path.isfile(path):
                os.unlink(path)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(CODE_DIR, "config.yaml"), help="yaml config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    app = create_app(cfg)

    server_cfg = cfg.get("server") or {}
    uvicorn.run(app, host=server_cfg.get("host", "0.0.0.0"), port=server_cfg.get("port", 50001))


if __name__ == "__main__":
    main()
