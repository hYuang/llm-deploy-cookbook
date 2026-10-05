# llm-deploy-cookbook

模型部署实战手册（cookbook）。每个模型一个 recipe，记录**我们实际部署时**的环境、代码、踩坑与解法，而不是照搬官方 README。

## 目录结构约定

每个 recipe 固定如下结构，方便复用与检索：

```
recipes/<model>/
├── README.md             # 概览：官方资源（仓库/模型/论文）/硬件信息/部署步骤
├── env/
│   └── requirements.txt  # 我们在上游依赖之外额外固定的依赖
├── code/
│   ├── infer.py          # 本地推理参考脚本
│   └── server.py         # 服务化（FastAPI）参考脚本
└── notes/
    └── pitfalls.md       # 实际部署踩到的坑 + 解法（持续补充）
```

## Recipes

| 模型 | 类型 | 状态 | 说明 |
| --- | --- | --- | --- |
| [CosyVoice3](recipes/cosyvoice3/README.md) | TTS（0.5B） | ✅ 已部署 | 零样本语音克隆 / 跨语言 / 指令控制 |

## 如何新增一个 recipe

1. `cp -r recipes/cosyvoice3 recipes/<new-model>` 作为模板
2. 按上面的结构填充（env / code / notes）
3. 在本文件 Recipes 表格里加一行
