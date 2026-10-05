# CosyVoice3 FastAPI 服务（参考上游 runtime/python/fastapi/server.py 的精简自包含版）
#
# 部署：app.py / config.yaml / service/ 拷到 CosyVoice 仓库根目录
#   conda activate cosyvoice
#   cd <app.py 所在目录>
#   python app.py                     # 不带参数：读同目录 config.yaml
#   python app.py --config xxx.yaml   # 或指定配置文件
#
# 配置全部读自 yaml（见 config.yaml）：模型从本地加载，路径在 yaml 中配置
#
# 输出: 22050Hz / 16bit / mono wav 文件
#
# 注意: prompt_wav 为音色上传，服务端把上传的音频存到本地临时目录再喂给模型；
# 不传则用 yaml 配置的默认参考音频。

import argparse
import os
import sys
import tempfile

import uvicorn
import yaml

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=os.path.join(CODE_DIR, "config.yaml"), help="yaml config path")
args = parser.parse_args()


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not (cfg.get("model") or {}).get("model_dir"):
        sys.exit("config missing model.model_dir")
    return cfg


def resolve_path(p: str, base: str) -> str:
    # 绝对路径直接用，相对路径相对 base 解析
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(base, p))


config = load_config(args.config)

# app.py 和 CosyVoice 仓库根目录放一起，仓库就是脚本所在目录
repo_dir = CODE_DIR

# Matcha-TTS 子模块必须已检出，否则 import cosyvoice 会报 No module named 'matcha'
MATCHA_DIR = os.path.join(repo_dir, "third_party", "Matcha-TTS")
if not os.path.isdir(os.path.join(MATCHA_DIR, "matcha")):
    sys.exit(
        f"matcha 包未找到: {MATCHA_DIR}/matcha\n"
        f"修复: cd {repo_dir} && git submodule update --init --recursive\n"
        "（CosyVoice 仓库需带 --recursive 克隆；若该目录非 git 仓库，需单独补 Matcha-TTS 子模块）"
    )
sys.path.append(repo_dir)
sys.path.append(MATCHA_DIR)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

from service.api import create_app  # noqa: E402

# 本地模型目录：绝对路径直接用，相对路径相对 repo_dir 解析
model_dir = resolve_path(config["model"]["model_dir"], repo_dir)

# 默认参考音频：绝对路径直接用，相对路径相对 config.yaml 所在目录（code/）解析
# 支持单值（所有端点共用）或 {zero_shot, cross_lingual} 两种写法
prompt_wav_cfg = config["model"].get("prompt_wav") or "asset/zero_shot_prompt.wav"
if isinstance(prompt_wav_cfg, str):
    prompt_wav_cfg = {"zero_shot": prompt_wav_cfg}
default_prompt_wav = {
    "zero_shot": resolve_path(prompt_wav_cfg.get("zero_shot") or "asset/zero_shot_prompt.wav", CODE_DIR),
    "cross_lingual": resolve_path(
        prompt_wav_cfg.get("cross_lingual") or "asset/cross_lingual_prompt.wav", CODE_DIR
    ),
}

# 音色上传落盘目录：绝对路径直接用，相对路径相对 repo_dir 解析；默认系统临时目录
upload_dir_cfg = config.get("upload_dir")
upload_dir = (
    resolve_path(upload_dir_cfg, repo_dir)
    if upload_dir_cfg
    else os.path.join(tempfile.gettempdir(), "cosyvoice_uploads")
)
os.makedirs(upload_dir, exist_ok=True)

model = AutoModel(model_dir=model_dir)

app = create_app(model, default_prompt_wav, upload_dir)

server_cfg = config.get("server") or {}
host = server_cfg.get("host", "0.0.0.0")
port = server_cfg.get("port", 50000)


if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
