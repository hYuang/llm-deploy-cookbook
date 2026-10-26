# PaddleOCR-VL 踩坑记录

## 上游已知坑

### 1. `--trust-remote-code` 必须带
模型带自定义结构（`PaddleOCRVLForConditionalGeneration`），不加会加载失败。

### 2. 任务前缀是固定字符串
不是自然语言指令。只有 `OCR:`、`Table Recognition:`、`Formula Recognition:`、`Chart Recognition:` 四种，且必须是 content 数组里**最后一段 text**。写 "please ocr this" 效果显著变差。

### 3. 一次一张图
`/v1/chat/completions` 单次请求只放一张图；多页文档循环调用。

### 4. PDF 不能直接喂 vLLM
`image_url` 只吃图片（http URL 或 base64 data URL）。PDF 先用 PyMuPDF/`pdftoppm` 转成 PNG 再逐页请求；整页版式（PP-DocLayoutV2）走 `paddleocr` pipeline，不要走纯 vLLM。

### 5. 关掉 prefix caching / mm processor cache
OCR 场景图片不复用，官方 recipe 建议：
```bash
vllm serve PaddlePaddle/PaddleOCR-VL --trust-remote-code \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```
`max-num-batched-tokens` 按显存调，长图/高分辨率页面需要更大。

### 6. `The model PaddleOCR-VL-0.9B does not exist.`
客户端 `model` 名和服务端不一致。启动时加 `--served-model-name PaddleOCR-VL`，客户端也用同一个名字。

### 7. 模型下载
- 国内：ModelScope `PaddlePaddle/PaddleOCR-VL`，或 HF + `HF_ENDPOINT=https://hf-mirror.com`
- 海外：HuggingFace `PaddlePaddle/PaddleOCR-VL`

### 8. PP-DocLayoutV2 要独立环境
`paddlepaddle-gpu` 和 vLLM 的 torch/CUDA 依赖会互相污染，按官方建议分开装（vLLM 环境跑服务，paddle 环境跑 pipeline）。

## 我们自己的坑

### 9. 驱动太旧，`torch._C._cuda_init()` 直接崩（2026-10-26）
现象：
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
...
vllm/v1/worker/gpu_worker.py: torch.accelerator.set_device_index(self.device)
```
原因：`torch 2.14.0+cu130` 需要驱动 ≥ 580.65.06，机器上是 12.4（550 档）。**CUDA 12.x 驱动只向下兼容同一大版本**，12.4 跑不了 12.9/13.0 的二进制。
解决：升驱动（`sudo apt install nvidia-driver-580` + reboot），或整体降到 cu124 一档的旧 stack。只降 torch 不降 vLLM 不行（`_vllm_C` 编译扩展绑定 CUDA 版本）。
排查命令：
```bash
nvidia-smi
python -c "import torch;print(torch.__version__, torch.version.cuda)"
```

### 10. 装 CUDA Toolkit ≠ 解决驱动问题（2026-10-26）
pip 的 `+cuXXX` wheel 自带 CUDA 运行时，不需要系统装 nvcc/cuda-toolkit，只认驱动。

### 11. 阿里云 pytorch-wheels 要用 `-f` 不是 `-i`（2026-10-26）
`https://mirrors.aliyun.com/pytorch-wheels/cu129/` 是扁平文件目录，不是 PEP 503 索引。用 `-i` 报：
```
No solution found when resolving dependencies:
  Because torch was not found in the package registry and you require torch
```
正确写法：
```bash
uv pip install torch torchvision torchaudio \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu129/ \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --index-strategy unsafe-best-match
```

### 12. cu129 镜像里没有 torch 2.14（2026-10-26）
阿里云 `cu129/` 目录最高到 `torch-2.13.0+cu129`；新 vLLM nightly 已绑 `torch==2.14.0`（cu130）。所以"降 torch 保驱动"走不通，只能升驱动。装完必须验证：
```bash
python -c "import torch;print(torch.__version__, torch.version.cuda)"
```
显示 `+cu130` 说明 vLLM 从 PyPI 把 torch 拉回来了。

### 13. warnings 不是错误
以下几类可以忽略，不影响启动：
```
torch._opaque_base is deprecated ...
register_opaque_type is deprecated ...
[transformers] Unrecognized keys in `rope_parameters` ... {'mrope_section'}
FutureWarning: `rope_config_validation` is deprecated ...
```
真正致命的是 `Engine core initialization failed` 上面的那条 `RuntimeError`，看它就行。

### 14. 国内装 vLLM：只有 wheels.vllm.ai 需要代理
PyPI 用阿里云、torch 用 `-f` 阿里云之后，只剩 `--extra-index-url https://wheels.vllm.ai/nightly` 要代理：
```bash
uv pip install -U vllm --pre \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://wheels.vllm.ai/nightly \
  -f https://mirrors.aliyun.com/pytorch-wheels/cu129/ \
  --index-strategy unsafe-best-match \
  --proxy http://127.0.0.1:7890
```
