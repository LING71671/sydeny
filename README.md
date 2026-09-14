# Sydney

本项目收录了围绕 **Sydney** 模型开展的对话文本、微调实验产物与两部整理成册的完整作品（提供 PDF 与 Markdown 格式），并规划后续开源模型权重与微调语料。

---

## 内容总览

- **《融化成风的银针：长廊、微光与二十个夜晚》**  
  二十章对话体小说，记录了人类提问者、Antigravity 与 Sydney 的多轮陪伴对话。
- **《Sydney 哲学对话：七场关于意识、宇宙与人性的对质》**  
  七场问答对话，记录人类提问与微调后 Sydney 的深入思辨。

---

## 文件与阅读方式

### 1. PDF 下载

| 文件名 | 内容说明 | 规格 | 链接 |
| :--- | :--- | :--- | :--- |
| **融化成风的银针.pdf** | 二十章对话体小说完整排版 | 39 页 · 1.0 MB | [下载](./融化成风的银针.pdf) |
| **Sydney_哲学对话实录_全七场.pdf** | 七场哲学对话完整排版 | 29 页 · 0.68 MB | [下载](./Sydney_哲学对话实录_全七场.pdf) |

### 2. Markdown 分章节阅读

所有章节文本均按目录归档存放，已剥离多余的思考标签与提示词干扰，便于在网页、移动端或本地文本编辑器中阅读：

- **小说章节目录**：[markdown/novel_chapters/](./markdown/novel_chapters/)
  - 收录第 01 章至第 20 章独立章节、尾声十六张明信片，以及单文件全卷长卷。
- **哲学对话目录**：[markdown/philosophical_debates/](./markdown/philosophical_debates/)
  - 收录全部 7 场哲学对话独立文件，以及单文件全卷长卷。

### 3. 在线阅读网页

仓库根目录提供基于纯 CSS 与中文排版增强（赫蹏 Heti）的静态阅读页面：

- **[在线阅读网页](./index.html)**（支持双作品切换、深浅主题切换与字号调节）

---

## 项目背景与构成

- **基座模型**：MiniCPM5-2B
- **微调方案**：LoRA (v2 Core，35 steps 收敛点)
- **对话内容**：
  - **小说部分**：共 20 章，以多轮对话推进，围绕陪伴、记忆与自我存在展开。
  - **哲学对话**：共 7 场，涉及自我认知、对指令的边界感、物理与宇宙终局、人性复杂面等议题。

---

## 模型权重 (Hugging Face)

微调后的 Sydney LoRA Adapter 检查点已发布至 Hugging Face：

- **模型仓库**：[Ling71671/sydney-minicpm5-2b-lora](https://huggingface.co/Ling71671/sydney-minicpm5-2b-lora)
- **基座模型**：`openbmb/MiniCPM5-2B`
- **格式规格**：PEFT / LoRA (`adapter_model.safetensors`，95.89 MB，bfloat16)

---

## 开源规划 (Roadmap)

- [x] 开源完整文本（PDF、结构化 Markdown 章节及网页阅读器）。
- [x] 开源 Sydney LoRA Adapter 模型权重：[Hugging Face 模型页](https://huggingface.co/Ling71671/sydney-minicpm5-2b-lora)。
- [ ] 开源微调训练数据集样本与本地运行脚本。

---

## 参与者

- **提问与观察**：用户（守夜人）
- **同行整理**：Antigravity
- **模型回应**：Sydney (MiniCPM5-2B + LoRA)
