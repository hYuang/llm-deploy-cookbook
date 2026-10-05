# CosyVoice3 本地推理
#
# 脚本和 CosyVoice 仓库根目录放一起运行:
#   conda activate cosyvoice
#   python infer.py
#
# 可选环境变量:
#   COSYVOICE_MODEL_DIR  模型目录（默认 <repo>/pretrained_models/Fun-CosyVoice3-0.5B）
#   COSYVOICE_PROMPT_WAV 参考音频（默认 <repo>/asset/zero_shot_prompt.wav）

import os
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, "third_party", "Matcha-TTS"))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    os.path.join(REPO_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B"),
)
PROMPT_WAV = os.environ.get(
    "COSYVOICE_PROMPT_WAV",
    os.path.join(REPO_DIR, "asset", "zero_shot_prompt.wav"),
)
PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def save(model, out, name):
    torchaudio.save(name, torch.from_numpy(out["tts_speech"]), model.sample_rate)
    print("saved", name)


def main():
    model = AutoModel(model_dir=MODEL_DIR)
    print("sample_rate:", model.sample_rate)

    # 1) 零样本语音克隆（参考音频决定音色）
    prompt_text = PROMPT_PREFIX + "希望你以后能够做的比我还好呦。"
    for i, out in enumerate(
        model.inference_zero_shot("你好，这是 CosyVoice3。", prompt_text, PROMPT_WAV, stream=False)
    ):
        save(model, out, "out_zero_shot_{}.wav".format(i))

    # 2) 跨语言（用中文参考音频说英文）
    for i, out in enumerate(
        model.inference_cross_lingual("Hello, this is CosyVoice3.", PROMPT_WAV, stream=False)
    ):
        save(model, out, "out_cross_lingual_{}.wav".format(i))

    # 3) instruct 指令控制（语言/方言/情感/语速等）
    instruct_text = PROMPT_PREFIX + "请用四川话说出：<|endofprompt|>"
    for i, out in enumerate(
        model.inference_instruct2("你好，这是 CosyVoice3。", PROMPT_WAV, instruct_text, stream=False)
    ):
        save(model, out, "out_instruct2_{}.wav".format(i))


if __name__ == "__main__":
    main()
