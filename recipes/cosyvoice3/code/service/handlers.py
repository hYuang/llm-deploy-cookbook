# CosyVoice3 服务路由处理（由 app.py 拆分而来）
# 输出 wav：模型 stream=True 逐块产出，攒齐后拼成单个 wav 文件返回
# prompt_wav 为音色上传：服务端把上传的音频存到本地临时目录再喂给模型；不传则用 default_prompt_wav

import io
import os
import uuid
import wave
from datetime import datetime

import numpy as np
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, Response


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

    return router
