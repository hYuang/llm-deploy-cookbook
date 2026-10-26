# PaddleOCR-VL 部署

PaddleOCR-VL-0.9B，文档解析专用 VLM：NaViT 风格动态分辨率视觉编码器 + ERNIE-4.5-0.3B 语言模型，支持 OCR、表格、公式、图表识别。官方推荐用 vLLM 部署，服务端自带 OpenAI 兼容接口。

## 官方资源

- 模型（HuggingFace）：<https://huggingface.co/PaddlePaddle/PaddleOCR-VL>
- 模型（ModelScope，国内）：`PaddlePaddle/PaddleOCR-VL`
- vLLM 官方 recipe：<https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html>
- PaddleOCR 仓库（PP-DocLayoutV2 整页解析）：<https://github.com/PaddlePaddle/PaddleOCR>

## 服务器硬件信息

> 待补充

| 项 | 信息 |
| --- | --- |
| GPU（型号/数量/显存） | |
| 驱动 / CUDA / cuDNN | |
| 操作系统 | |
| CPU / 内存 / 磁盘 | |

## 前置：驱动版本（最容易卡住的地方）

vLLM + torch 的 wheel 绑定 CUDA 版本，**驱动太旧会在 `torch._C._cuda_init()` 直接崩**，报：

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
```

驱动与 CUDA 运行时对应关系（Linux）：

| CUDA 运行时 | 最低驱动 |
| --- | --- |
| 12.4 | 550.54.14 |
| 12.8 | 570.26 |
| 12.9 | 575.51.03 |
| 13.0 | 580.65.06 |

先确认：

```bash
nvidia-smi                                  # 右上角 CUDA Version
python -c "import torch;print(torch.version.cuda)"
```

两者必须匹配（`torch.version.cuda` 不能高于 `nvidia-smi` 显示的值）。**装 CUDA Toolkit 没用**，pip 的 wheel 已自带 CUDA 运行时，只认驱动。

驱动不够就先升：

```bash
ubuntu-drivers devices
sudo apt install -y nvidia-driver-580      # 或 ubuntu-drivers autoinstall
sudo reboot
```

## 部署步骤

### 1. 环境与依赖

模型需要较新的 vLLM（PaddleOCR-VL 架构目前只有 nightly 支持；正式版 ≥ 0.11.1 后可直接用正式版）。

```bash
cd /home/ubuntu/script
uv venv --python 3.12
source .venv/bin/activate

# 官方源（能直连海外时最省事）
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

国内镜像版（PyPI 走阿里云，torch 目录用 `-f`，它**不是** PEP 503 索引，用 `-i` 会报 `torch was not found`）：

```bash
uv pip install -U vllm --pre \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/nightly \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu129/ \
  --index-strategy unsafe-best-match
```

`wheels.vllm.ai` 没有国内镜像，不通就加 `--proxy http://127.0.0.1:7890`（只影响这一个源）。

校验：

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda)"
python -c "import vllm;print(vllm.__version__)"
```

### 2. 下载模型

```bash
# HuggingFace（国内走 hf-mirror）
export HF_ENDPOINT=https://hf-mirror.com
hf download PaddlePaddle/PaddleOCR-VL --local-dir ./PaddleOCR-VL

# 或 ModelScope
uv pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
modelscope download --model PaddlePaddle/PaddleOCR-VL --local_dir ./PaddleOCR-VL
```

### 3. 启动 vLLM

```bash
# 直接拉 HF 仓库
vllm serve PaddlePaddle/PaddleOCR-VL \
  --served-model-name PaddleOCR-VL \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0

# 或用本地目录（离线 / 内网）
vllm serve ./PaddleOCR-VL \
  --served-model-name PaddleOCR-VL \
  --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

- `--trust-remote-code` 必须带，否则模型加载失败。
- `--no-enable-prefix-caching` + `--mm-processor-cache-gb 0`：OCR 场景图片不复用，关掉省显存和哈希开销。
- `--max-num-batched-tokens` 按显存调，太小会限制长图吞吐。
- `--served-model-name` 统一模型名，否则客户端要填完整路径/仓库名。

服务默认监听 `:8000`，OpenAI 兼容端点为 `/v1/models`、`/v1/chat/completions`。

### 4. OpenAI 兼容调用

图片放 `image_url`（http URL 或 `data:image/...;base64,...`），**任务前缀必须是最后一段 text**：

| task | 前缀 |
| --- | --- |
| OCR | `OCR:` |
| 表格 | `Table Recognition:` |
| 公式 | `Formula Recognition:` |
| 图表 | `Chart Recognition:` |

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PaddleOCR-VL",
    "temperature": 0,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {"type": "text", "text": "OCR:"}
      ]
    }]
  }'
```

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=3600)

b64 = base64.b64encode(open("doc.png", "rb").read()).decode()
resp = client.chat.completions.create(
    model="PaddleOCR-VL",
    temperature=0.0,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "OCR:"},
    ]}],
)
print(resp.choices[0].message.content)
```

一次请求只放一张图；多页/多图循环调用。PDF 不能直接喂给 vLLM，先转成图片。

### 5. 可选：包装服务（multipart 上传 + 自动补任务前缀）

`vllm serve` 已经够用，以下脚本只是把常用操作包一层，适合不想在客户端处理 base64 的场景（`code/server.py`，参考 `../env/requirements.txt` 安装依赖）：

```bash
cd <cookbook>/recipes/paddleocr-vl/code
pip install -r ../env/requirements.txt
python server.py                      # 读同目录 config.yaml
python server.py --config xxx.yaml
```

配置见 `code/config.yaml`（后端地址、模型名、默认任务、可选版面解析、host/port）。

端点：

| 端点 | 说明 |
| --- | --- |
| `GET /v1/models` | OpenAI 模型列表 |
| `POST /v1/chat/completions` | OpenAI 兼容透传；最后一段 text 不是任务前缀时自动补 `defaults.task` |
| `POST /ocr` | multipart：`file`（图片）+ `task`（`ocr`/`table`/`formula`/`chart`），返回纯文本 |
| `POST /parse` | 可选：PP-DocLayoutV2 整页解析，返回 `{pages, markdown}`（需 `pipeline.enabled=true`） |
| `GET /health` | 健康检查 |

```bash
curl -s -X POST "http://127.0.0.1:50001/ocr" \
  -F "file=@./doc.png" -F "task=table"
```

### 6. 可选：整页版面解析（PP-DocLayoutV2）

只用 vLLM 是"图片 → 文本"，官方完整文档解析还要 PP-DocLayoutV2 做版面检测。**建议和 vLLM 用独立环境**，避免 paddlepaddle 与 torch 互相污染：

```bash
uv pip install paddlepaddle-gpu==3.2.1 --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/
uv pip install -U "paddleocr[doc-parser]"
uv pip install safetensors
```

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:8000/v1",
    layout_detection_model_name="PP-DocLayoutV2",
    layout_detection_model_dir="/path/to/PP-DocLayoutV2",
)
for i, res in enumerate(pipeline.predict("doc.png")):
    res.save_to_json(save_path=f"output_{i}.json")
    res.save_to_markdown(save_path=f"output_{i}.md")
```

包装服务里打开 `pipeline.enabled: true` 并配好 `layout_detection_model_dir`，即可走 `POST /parse`。

## 目录

- `env/requirements.txt` — 包装服务额外依赖（vLLM/torch 由 README 的安装命令管理，不写死在此）
- `code/config.yaml` — 后端地址、模型名、默认任务、可选版面解析、host/port
- `code/server.py` — FastAPI 包装服务（OpenAI 兼容 + 上传 + 可选整页解析）
- `notes/pitfalls.md` — 踩坑记录（驱动版本、镜像源、任务前缀等）
