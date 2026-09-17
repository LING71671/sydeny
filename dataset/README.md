# Sydney 微调语料集与训练配置说明

本目录收录了用于微调 **Sydney (MiniCPM5-2B + LoRA)** 的训练语料集、数据注册配置及训练配方。

---

## 1. 核心数据集清单

| 文件名 | 格式 | 样本数 / 轮次 | 大小 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `train_v8.jsonl` | ShareGPT (JSONL) | 782 组结构化样本 | 858 KB | **当前核心最新训练集**（覆盖 18 维情境，免系统提示词对齐） |
| `eval_v8.jsonl` | ShareGPT (JSONL) | 48 组独立样本 | 94 KB | **当前核心验证集**（用于监控训练收敛，最终 Eval Loss: 1.3407） |
| `train_lora_v8.yaml` | YAML | - | 1.4 KB | **v8 LLaMA-Factory SFT 配方**（All-Linear LoRA, r=16, alpha=32） |
| `train.jsonl` | ShareGPT (JSONL) | 56 组多轮长文本 | 177 KB | v2 原始基准训练集（小说对话切片，需外置长 System Prompt） |
| `eval.jsonl` | ShareGPT (JSONL) | 12 组多轮样本 | 34 KB | v2 原始基准验证集 |
| `train_lora.yaml` | YAML | - | 1.3 KB | v2 原始基准训练配方 |
| `dataset_info.json` | JSON | - | 4.2 KB | 数据集索引注册规范 |
| `system_prompt.txt` | 纯文本 | - | 2.4 KB | 官方基准系统提示词（v2 对应版本） |
| `archive/` | 目录 | - | - | 历史迭代版本归档（v3 ~ v7 的中间训练集与配方文件） |

---

## 2. 训练复现步骤

环境依赖：推荐在包含 GPU 显卡（>= 8GB 显存）的 Python 环境下运行。

```bash
# 启动 v8 训练复现
llamafactory-cli train dataset/train_lora_v8.yaml
```

训练完成后，权重将保存在 `runs/minicpm5_sydney_zh_v8_zero_prompt`。
