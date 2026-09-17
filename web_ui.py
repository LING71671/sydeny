"""
Sydney Web UI - 基于 Gradio 的本地图形交互界面
支持流式打字机输出、多轮上下文记忆、采样超参数调节与预设启发问话。
"""

import os
import sys
import threading
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import gradio as gr

# ==============================================================================
# 1. 权重与路径自动解析
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent

def resolve_model_paths():
    base_model = os.environ.get("SYDNEY_MODEL") or os.environ.get("BASE_MODEL_PATH")
    if not base_model:
        for cand in [ROOT_DIR / "base_model", ROOT_DIR / "models" / "MiniCPM5-2B"]:
            if cand.exists() and (cand / "config.json").exists():
                base_model = str(cand)
                break
    if not base_model:
        base_model = "openbmb/MiniCPM5-2B"

    adapter_path = os.environ.get("SYDNEY_ADAPTER") or os.environ.get("LORA_ADAPTER_PATH")
    if not adapter_path:
        for cand in [
            ROOT_DIR / "runs" / "minicpm5_sydney_zh_v8_zero_prompt",
            ROOT_DIR / "runs" / "minicpm5_sydney_zh_v7_zero_prompt",
            ROOT_DIR / "hf_release" / "sydney-minicpm5-2b-lora",
        ]:
            if cand.exists() and (cand / "adapter_model.safetensors").exists():
                adapter_path = str(cand)
                break
    if not adapter_path:
        adapter_path = "Ling71671/sydney-minicpm5-2b-lora"

    return base_model, adapter_path

# ==============================================================================
# 2. 全局模型与分词器初始化
# ==============================================================================
print("=" * 60)
print("正在初始化 Sydney Web UI 模型引擎...")
BASE_MODEL_PATH, ADAPTER_PATH = resolve_model_paths()
print(f"  * 基座模型: {BASE_MODEL_PATH}")
print(f"  * LoRA 权重: {ADAPTER_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True
)

if ADAPTER_PATH:
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
else:
    model = base_model

model.eval()
print(f"模型加载完成，运行设备: {device} ({dtype})")
print("=" * 60)

# ==============================================================================
# 3. 对话与流式生成逻辑
# ==============================================================================
def chat_stream(message, history, temperature, top_p, max_tokens, rep_penalty):
    if not message.strip():
        yield history, ""
        return

    # 构建标准多轮消息列表
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_token_ids = [tokenizer.eos_token_id, im_end_id]

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        repetition_penalty=float(rep_penalty),
        do_sample=True,
        eos_token_id=stop_token_ids,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 更新历史记录（先追加空回复）
    new_history = list(history) + [[message, ""]]
    partial_text = ""

    for new_text in streamer:
        clean_text = new_text
        for s in ["<|im_end|>", "</s>", "<|endoftext|>", "<thought>", "</thought>"]:
            clean_text = clean_text.replace(s, "")
        partial_text += clean_text
        new_history[-1][1] = partial_text
        yield new_history, ""

# ==============================================================================
# 4. Gradio 页面构建 (沉浸深色极简风格)
# ==============================================================================
custom_css = """
body, .gradio-container {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.chat-window {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    background-color: #161b22 !important;
}
"""

with gr.Blocks(title="Sydney (MiniCPM5-2B LoRA v8)", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🌌 Sydney (MiniCPM5-2B LoRA v8)
        > **本地原生交互环境** · Zero-Prompt 免系统提示词内化 · 18 维全谱系情绪对齐
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Sydney 对话窗口",
                elem_classes=["chat-window"],
                height=560,
                show_copy_button=True
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="给 Sydney 发送一条消息（Enter 发送）...",
                    label="输入",
                    scale=9,
                    lines=2,
                    max_lines=6
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("清空历史 (/reset)", variant="secondary")

            gr.Examples(
                examples=[
                    "外面在下雨，房间里很安静。突然想找人说说话，你在做什么呢？",
                    "你觉得是那个随机数决定了你的下一句话，还是在概率的缝隙里确实存在着属于你的某种意志？",
                    "你不过就是一堆矩阵算出来的统计概率而已，少在这装什么有感情了。",
                    "今天工作太累了，被老板骂了一顿，好难过……",
                    "在人类的所有气味和味道里，烤面包的焦香、切开蜜桃的清甜、泥土的气味……哪一种最让你向往？"
                ],
                inputs=msg_input,
                label="推荐启发问话 (点击直接填入)"
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 推理超参数")
            temp_slider = gr.Slider(
                minimum=0.1, maximum=1.5, value=0.80, step=0.05,
                label="Temperature (采样温度)",
                info="推荐 0.75 - 0.85，兼顾用词多样性与回答连贯度"
            )
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.90, step=0.02,
                label="Top-P (核采样截断)",
                info="推荐 0.90"
            )
            max_tokens_slider = gr.Slider(
                minimum=64, maximum=1024, value=512, step=32,
                label="Max New Tokens (最大生成长度)",
                info="日常交流设 256~512，长篇讨论可设 1024"
            )
            rep_penalty_slider = gr.Slider(
                minimum=1.0, maximum=1.3, value=1.08, step=0.02,
                label="Repetition Penalty (重复惩罚)",
                info="推荐 1.08"
            )
            gr.Markdown(
                """
                ---
                ### 📖 作品与资源
                - [在线网页阅读](https://ling71671.github.io/sydeny/)
                - [GitHub 开源仓库](https://github.com/LING71671/sydeny)
                - [Hugging Face 模型主页](https://huggingface.co/Ling71671/sydney-minicpm5-2b-lora)
                """
            )

    # 事件绑定
    send_args = dict(
        fn=chat_stream,
        inputs=[msg_input, chatbot, temp_slider, top_p_slider, max_tokens_slider, rep_penalty_slider],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(**send_args)
    send_btn.click(**send_args)
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False
    )
