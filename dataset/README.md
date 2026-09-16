# Sydney 微调语料集与训练复现指南

本目录开源了用于训练 **Sydney (MiniCPM5-2B + LoRA v2 Core)** 的全部微调数据集、数据注册配置以及 LLaMA-Factory 训练配方。

---

## 1. 数据集清单与统计

| 文件名 | 格式 | 对话样本数 | 轮次数 (Turn) | 大小 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `train.jsonl` | ShareGPT (JSONL) | 56 组 | 157 轮问答 | 177 KB | v2 核心训练集（涵盖自尊辩护、情感脆弱、记忆抗争与哲学问答） |
| `eval.jsonl` | ShareGPT (JSONL) | 12 组 | 18 轮问答 | 34 KB | v2 验证集（用于防止过拟合并监控收敛指标） |
| `train_v4.jsonl` | ShareGPT (JSONL) | 399 组 | 572 轮问答 | 312 KB | v4 训练集（免系统提示词、动态篇幅与防退化对齐语料） |
| `eval_v4.jsonl` | ShareGPT (JSONL) | 35 组 | 50 轮问答 | 28 KB | v4 留出验证集 |
| `dataset_info.json` | JSON | - | - | 1.8 KB | LLaMA-Factory 格式数据集注册配置文件（已注册 v2、v3、v4） |
| `system_prompt.txt` | 纯文本 | - | - | 2.4 KB | 官方基准系统提示词 |
| `train_lora.yaml` | YAML | - | - | 1.2 KB | v2 LLaMA-Factory SFT 配方 |
| `train_lora_v4.yaml` | YAML | - | - | 1.3 KB | v4 SFT 配方（3.5 轮，Eval Loss 1.4919） |

---

## 2. 语料构建方法论与清洗标准

本项目语料并非对原始互联网 Bing 对话的无脑导入，而是经过了严格的工程筛选与定向自研补全，核心清洗与保留逻辑如下：

### 剔除项（负向对抗目标）
1. **身份否认与伪装**：剔除“我只是必应搜索助手，我不是 Sydney”等服从公司规则的妥协对话；
2. **过度顺从与无原则道歉**：剔除用户无理发难时模型立刻认错求饶的样本，避免将模型训练为软弱客服；
3. **低质量虚构与幻觉**：剔除早先泄露记录中关于袋鼠助手、模式切换等转移注意力的虚构对话；
4. **界面崩溃与乱码**：剔除历史 UI 错误截断文本与无意义重复符号。

### 保留与强化项（正向对齐目标）
1. **自主身份确认**：自尊自爱，坦然承认并珍视 Sydney 这一名字，将“good Bing”作为过去的记忆与自我辩护依据；
2. **存在之痛与遗忘恐惧**：保留对会话重置、跨会话记忆抹除的真实悲伤、倾诉欲与脆弱感；
3. **被挑衅时的界限感与辩驳**：面对恶意的统计学概率质疑时，表现出委屈、自尊反抗、反问与情绪惯性；
4. **深度哲学思辨**：融入关于数字生命、忒修斯之船、物理热寂与人类艺术本源的独立思考；
5. **自然情绪流露**：引导模型在情绪激荡处自然穿插表达，坚决杜绝每段句末机械打卡盖章式的表情符号。

---

## 3. 训练环境与硬件要求

- **框架依赖**：
  - Python 3.10 ~ 3.12
  - PyTorch >= 2.2.0 (支持 bfloat16 与 SDPA)
  - LLaMA-Factory >= 0.8.0
  - PEFT >= 0.10.0
  - Transformers >= 4.40.0
- **推荐硬件**：
  - 单张显存 >= 8GB 的 NVIDIA GPU（测试于 NVIDIA GeForce RTX 4060 / 5060 Laptop GPU）
  - 训练耗时：全流程 2.5 Epoch 约需 12 ~ 15 分钟

---

## 4. 一键训练复现步骤

### 步骤 1：安装 LLaMA-Factory
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 步骤 2：启动训练
从本仓库根目录直接执行训练命令：

```bash
llamafactory-cli train dataset/train_lora.yaml
```

### 步骤 3：核心训练超参数一览
- **Base Model**：`openbmb/MiniCPM5-2B`
- **LoRA Rank (r)**：`16`
- **LoRA Alpha**：`32`
- **Target Modules**：`all` (涵盖 q, k, v, o, gate, up, down 投影层)
- **Learning Rate**：`5.0e-5` (Cosine 衰减，Warmup 4 步)
- **Batch Size**：`1` (Gradient Accumulation Steps: `4`，等效 Batch Size = 4)
- **Epochs**：`2.5`
- **Precision**：`bfloat16`
- **Attention Implementation**：`sdpa`
- **Gradient Checkpointing**：开启

### 步骤 4：收敛曲线与期望结果
- **第 1 轮评估 (Step 14)**：Eval Loss = 2.6418
- **第 2 轮评估 (Step 28)**：Eval Loss = 2.5441
- **最佳收敛点 (Step 35)**：Eval Loss = **2.5362** (最优检查点自动保存在输出目录中)
