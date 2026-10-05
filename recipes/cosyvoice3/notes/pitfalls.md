# CosyVoice3 踩坑记录

## 上游已知坑

### 1. 缺 third_party/Matcha-TTS 子模块
`git clone` 忘带 `--recursive`，`import cosyvoice` 报 `No module named 'matcha'`。
已克隆的仓库补一下：
```bash
cd CosyVoice
git submodule update --init --recursive
```

### 2. sox 缺失
librosa 依赖 sox，没装会报 `sox could not be found`。
```bash
sudo apt-get install -y sox libsox-dev
```

### 3. vLLM 与主环境冲突
vLLM 高吞吐需要独立 conda 环境（版本钉死）：
```bash
conda create -n cosyvoice-vllm -y python=3.10
conda activate cosyvoice-vllm
pip install vllm==0.11.0 transformers==4.57.1 numpy==1.26.4
```
主环境 `cosyvoice` 里不要装 vllm，会互相污染。

### 4. 模型下载
- 国内：ModelScope `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
- 海外：HuggingFace 同名
- 下载后目录名建议保持 `pretrained_models/Fun-CosyVoice3-0.5B`（或用 `COSYVOICE_MODEL_DIR` 指定）

### 5. 服务输出是裸 PCM，不是 wav
FastAPI 端点返回 22050Hz/16bit/mono 裸 PCM 字节流，直接存成 .wav 打不开。
```bash
ffmpeg -f s16le -ar 22050 -ac 1 -i out.pcm out.wav
```

### 6. prompt_text / instruct 必须带前缀
`You are a helpful assistant.<|endofprompt|>` 前缀不能少，否则效果差或报错。

### 7. prompt_wav 的上传方式
音色上传：`curl -F "prompt_wav=@./asset/zero_shot_prompt.wav"`，服务端把上传的音频存到本地临时目录（`upload_dir` 可配）再喂给模型，无需文件预先存在于服务端。不传则用 yaml 配置的默认参考音频。

### 8. 日语片假名
日语文本里片假名（外来词）发音可能不稳定，测试时留意。

### 9. ttsfrd 可选
上游 requirements 里 ttsfrd 编译失败可忽略，不影响主功能。

## 我们自己的坑（待补充）

> 部署过程中遇到的问题记录在这里：现象 / 原因 / 解决 / 日期。

- （待补充）
