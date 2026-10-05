# CosyVoice3 服务路由处理（由 app.py 拆分而来）
# 输出 wav：模型 stream=True 逐块产出，攒齐后拼成单个 wav 文件返回
# prompt_wav 为音色上传：服务端把上传的音频存到本地临时目录再喂给模型；不传则用 default_prompt_wav

import io
import os
import uuid
import wave
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# 默认参考文本（与 asset/zero_shot_prompt.wav 对应）；OpenAI 兼容接口未传 prompt_text 且 voice=default 时用
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"
DEFAULT_PROMPT_TEXT = PROMPT_PREFIX + "希望你以后能够做的比我还好呦。"


def wav_response(generator, sample_rate: int) -> Response:
    def to_int16(speech):
        if not isinstance(speech, np.ndarray):
            speech = speech.detach().cpu().numpy()
        return (np.clip(speech, -1.0, 1.0) * 32768.0).astype(np.int16).tobytes()

    frames = b"".join(to_int16(out["tts_speech"]) for out in generator)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(frames)
    return Response(content=buf.getvalue(), media_type="audio/wav")


def create_router(model, default_prompt_wav: dict, upload_dir: str) -> APIRouter:
    router = APIRouter()

    async def prompt_path(prompt_wav: UploadFile, default: str) -> str:
        if prompt_wav is None:
            return default
        data = await prompt_wav.read()
        if not data:
            return default
        ext = os.path.splitext(prompt_wav.filename or "prompt.wav")[1] or ".wav"
        saved = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
        with open(saved, "wb") as f:
            f.write(data)
        return saved

    @router.post("/inference_zero_shot")
    async def inference_zero_shot(tts_text: str, prompt_text: str, prompt_wav: UploadFile = File(None)):
        return wav_response(
            model.inference_zero_shot(tts_text, prompt_text, await prompt_path(prompt_wav, default_prompt_wav["zero_shot"]), stream=True),
            model.sample_rate,
        )

    @router.post("/inference_cross_lingual")
    async def inference_cross_lingual(tts_text: str, prompt_wav: UploadFile = File(None)):
        return wav_response(
            model.inference_cross_lingual(tts_text, await prompt_path(prompt_wav, default_prompt_wav["cross_lingual"]), stream=True),
            model.sample_rate,
        )

    @router.post("/inference_instruct2")
    async def inference_instruct2(tts_text: str, instruct_text: str, prompt_wav: UploadFile = File(None)):
        return wav_response(
            model.inference_instruct2(tts_text, await prompt_path(prompt_wav, default_prompt_wav["zero_shot"]), instruct_text, stream=True),
            model.sample_rate,
        )

    @router.get("/voices")
    async def list_voices():
        # 查看：列出已上传的音色（文件名 / 大小 / 修改时间）
        voices = []
        for name in sorted(os.listdir(upload_dir)):
            p = os.path.join(upload_dir, name)
            if os.path.isfile(p):
                st = os.stat(p)
                voices.append({
                    "name": name,
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                })
        return {"upload_dir": upload_dir, "voices": voices}

    @router.get("/voices/{name}")
    async def download_voice(name: str):
        # 下载：按文件名取回上传的音色文件
        base = os.path.basename(name)
        if base != name or not base:
            return Response(status_code=400)
        p = os.path.join(upload_dir, base)
        if not os.path.isfile(p):
            return Response(status_code=404)
        return FileResponse(p, filename=base, media_type="application/octet-stream")

    # ---- OpenAI 兼容接口（/v1/models、/v1/audio/speech）----

    class OpenAISTTSRequest(BaseModel):
        model: str = "cosyvoice-zero-shot"
        input: str
        voice: str = "default"
        response_format: str = "wav"
        speed: float = 1.0
        # 扩展字段（非 OpenAI 标准，可选）：音色参考文本，不传用默认
        prompt_text: Optional[str] = None

    @router.get("/v1/models")
    async def openai_list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "cosyvoice-zero-shot",
                    "object": "model",
                    "created": 0,
                    "owned_by": "cosyvoice",
                }
            ],
        }

    @router.post("/v1/audio/speech")
    async def openai_tts(req: OpenAISTTSRequest):
        if not req.input or not req.input.strip():
            raise HTTPException(status_code=400, detail="input must not be empty")
        if req.response_format != "wav":
            raise HTTPException(status_code=400, detail=f"unsupported response_format: {req.response_format}, only wav supported")
        voice = req.voice or "default"
        if voice == "default":
            prompt_wav = default_prompt_wav["zero_shot"]
            prompt_text = req.prompt_text or DEFAULT_PROMPT_TEXT
        else:
            base = os.path.basename(voice)
            if base != voice or not base:
                raise HTTPException(status_code=400, detail=f"invalid voice: {voice}")
            prompt_wav = None
            for name in (voice, *(f"{voice}{ext}" for ext in (".wav", ".mp3", ".flac", ".ogg"))):
                p = os.path.join(upload_dir, name)
                if os.path.isfile(p):
                    prompt_wav = p
                    break
            if prompt_wav is None:
                raise HTTPException(status_code=404, detail=f"voice not found: {voice}")
            prompt_text = req.prompt_text or PROMPT_PREFIX
        return wav_response(
            model.inference_zero_shot(req.input, prompt_text, prompt_wav, stream=True),
            model.sample_rate,
        )

    return router
