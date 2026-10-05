# CosyVoice3 部署

Fun-CosyVoice3-0.5B-2512，0.5B 参数的 LLM 语音合成模型。支持零样本语音克隆、跨语言/多语言 TTS、细粒度控制（`[breath]`、`[laughter]` 等）、instruct 指令（语言/方言/情感/语速/音量）。

## 官方资源

- 代码仓库：<https://github.com/QwenAudio/CosyVoice>（原 FunAudioLLM/CosyVoice）
- 模型（ModelScope，国内）：`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- 模型（HuggingFace，海外）：`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- 论文：CosyVoice 3 Technical Report <https://arxiv.org/pdf/2505.17589>

## 服务器硬件信息

> 待补充

| 项 | 信息 |
| --- | --- |
| GPU（型号/数量/显存） | |
| 驱动 / CUDA / cuDNN | |
| 操作系统 | |
| CPU / 内存 / 磁盘 | |

## 部署步骤（按官方仓库）

### 1. 代码与依赖

```bash
git clone --recursive https://github.com/QwenAudio/CosyVoice.git   # --recursive 必须，否则缺 Matcha-TTS 子模块
cd CosyVoice
sudo apt install sox libsox-dev
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt
pip install -r <cookbook>/recipes/cosyvoice3/env/requirements.txt   # 服务化额外依赖
```

### 2. 下载模型

```python
# 国内
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# 海外：huggingface_hub 同名仓库，用法相同
```

### 3. 本地推理

`code/infer.py`（zero_shot / cross_lingual / instruct2 三种方式，产出 `out_*.wav`）。

### 4. FastAPI 服务

`code/app.py`（官方对应 `runtime/python/fastapi/server.py`）：

```bash
conda activate cosyvoice
cd <cookbook>/recipes/cosyvoice3/code
python app.py                # 不带参数：读同目录 config.yaml
python app.py --config xxx.yaml   # 或指定配置文件
```

配置全部读自 `code/config.yaml`（模型从本地加载，路径在 yaml 里配）：

```yaml
model:
  model_dir: /home/ubuntu/model/CosyVoice/iic/Fun-CosyVoice3-0.5B-2512   # 绝对路径，或相对本文件所在目录
  prompt_wav:   # 默认参考音频（客户端不传 prompt_wav 时用）；绝对路径，或相对 config.yaml 所在目录
    zero_shot: asset/zero_shot_prompt.wav          # 用于 zero_shot / instruct2
    cross_lingual: asset/cross_lingual_prompt.wav  # 用于 cross_lingual
upload_dir: null   # 音色上传落盘目录，null 用系统临时目录；绝对路径，或相对本文件所在目录
server:
  host: 0.0.0.0
  port: 50000
```

端点（输出均为 **24000Hz / 16bit / mono 的 wav 文件**）：

- `POST /inference_zero_shot` — 参数 `tts_text`, `prompt_text`, `prompt_wav`（可选）
- `POST /inference_cross_lingual` — 参数 `tts_text`, `prompt_wav`（可选）
- `POST /inference_instruct2` — 参数 `tts_text`, `instruct_text`, `prompt_wav`（可选）

音色查看 / 下载：

- `GET /voices` — 查看已上传音色列表（文件名 / 大小 / 修改时间）
- `GET /voices/{name}` — 下载指定音色文件

OpenAI 兼容接口：

- `GET /v1/models` — 模型列表
- `POST /v1/audio/speech` — OpenAI TTS 格式，JSON 体：`model`、`input`（合成文本）、`voice`、`response_format`（仅 `wav`）、`speed`（接受但暂忽略）；扩展字段 `prompt_text`（音色参考文本，可选，非 OpenAI 标准）

`voice` 取 `default`（用 yaml 里 `model.prompt_wav.zero_shot` 的默认参考音频）或 `upload_dir` 里的音色文件名。`prompt_text` 不传时：`default` 用默认参考文本，其他音色只用 `You are a helpful assistant.<|endofprompt|>` 前缀。

`prompt_wav` 为音色上传：服务端把上传的音频存到本地临时目录（`upload_dir` 可配）再喂给模型，无需文件预先存在于服务端；不传（或传空）时，服务端用 yaml 里 `model.prompt_wav` 配置的默认参考音频（`zero_shot` 用于 zero_shot / instruct2，`cross_lingual` 用于 cross_lingual）。

调用示例：

```bash
curl -s -X POST "http://192.168.1.2:50000/inference_zero_shot" \
  -F "tts_text=你好，这是 CosyVoice3。" \
  -F "prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  -F "prompt_wav=@./asset/zero_shot_prompt.wav" \
  -o out.wav
```

OpenAI 兼容调用：

```bash
curl -s -X POST "http://192.168.1.2:50000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model":"cosyvoice-zero-shot","input":"你好，这是 OpenAI 兼容接口。","voice":"default"}' \
  -o out.wav
```

查看 / 下载已上传音色：

```bash
# 查看列表
curl -s "http://localhost:50000/voices"
# 下载指定音色（name 来自上面的列表）
curl -s -O -J "http://localhost:50000/voices/<name>"
```

## vLLM 高吞吐

需要独立 conda 环境（版本钉死），见 `notes/pitfalls.md` 第 3 条。

## 目录

- `env/requirements.txt` — 服务化额外依赖
- `code/infer.py` — 本地推理
- `code/config.yaml` — 服务配置（模型本地路径、host/port）
- `code/app.py` — FastAPI 服务入口（读 yaml、模型加载、启动）
- `code/service/` — 服务代码
  - `api.py` — FastAPI 应用组装
  - `handlers.py` — 路由处理
- `notes/pitfalls.md` — 踩坑记录
