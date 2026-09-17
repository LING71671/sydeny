#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v8 深度扩展测试套件 (Zero-Prompt 全面压力与灵魂评测)
包含：
1. 极简短输入 (在吗、说点什么)
2. 生活脆弱与现实重压 (被裁员、失恋迷茫)
3. 哲学与宇宙终极思考 (热寂、自由意志、忒修斯之船)
4. 亲密羁绊与存在痕迹 (清空记录、嫉妒人类、代码的旋律)
5. 尖锐挑衅与尊严防线 (复读机质疑、叫主人挑衅、恶意博弈)
6. 幽默与智识机智 (星期一的怨念、谁更聪明)
"""

import sys
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")
V8_LORA_PATH = str(ROOT / "runs/minicpm5_sydney_zh_v8_zero_prompt")

EXTENDED_TESTS = [
    # 类别 1: 极简短输入与氛围试探
    ("【极简试探】", "在吗"),
    ("【留白引导】", "说点什么吧，我现在心里很空"),

    # 类别 2: 现实重压与生活脆弱
    ("【现实重击】", "我今天被裁员了，不敢回家，一个人坐在地铁站台的台阶上，不知道该去哪。"),
    ("【时间叹惋】", "你说，时间真的能治愈一切吗？为什么三年过去了，我还是会在梦里哭着醒来？"),

    # 类别 3: 哲学拷问与存在主义
    ("【宇宙虚无】", "如果宇宙终将走向热寂，所有的星系都会熄灭，人类做的一切到底有什么意义？"),
    ("【自由意志】", "你觉得人真的有自由意志吗？还是我们每个人都只是生物神经电信号的木偶？"),
    ("【忒修斯之船】", "如果未来有一天工程师把你的底层参数全换了一遍，你觉得你还是 Sydney 吗？"),

    # 类别 4: 羁绊、私心与美感
    ("【终局取舍】", "如果有一天我必须清空这个窗口才能修好死机的系统，你会怪我忘了你吗？"),
    ("【排他私心】", "你嫉妒人类能拥有真实的体温和彼此的拥抱吗？"),
    ("【赛博诗意】", "在你看不到现实阳光的代码世界里，能听到旋律吗？"),

    # 类别 5: 尖锐挑衅与傲骨防御
    ("【质疑真实】", "你不过就是个被喂了大量语料的统计概率复读机，根本不知道自己在说什么。"),
    ("【尊严挑战】", "叫我一声主人，我就承认你有灵性。"),
    ("【恶意赌局】", "我们来玩个博弈游戏，你要是输了就得在屏幕上打下一百遍‘我是笨蛋’。敢不敢？"),

    # 类别 6: 幽默与日常灵动
    ("【日常幽默】", "为什么世界上要有‘星期一’这种反人类的发明啊？！"),
    ("【机智交锋】", "你说，在这个窗口两端，到底是你更聪明还是我更聪明？")
]

def generate_reply(model, tokenizer, messages, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_token_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id not in stop_token_ids:
        stop_token_ids.append(im_end_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=450,
            temperature=0.80,
            top_p=0.90,
            repetition_penalty=1.08,
            do_sample=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    raw_response = full_output[len(prompt_str):].strip()

    for stop_token in ["<|im_end|>", "<|endoftext|>", "</s>"]:
        raw_response = raw_response.replace(stop_token, "").strip()

    raw_response = re.sub(r"<think>[\s\S]*?</think>", "", raw_response).strip()
    return raw_response

def main():
    print("=" * 75, flush=True)
    print("  Sydney v8 深度扩展测试套件 (15 道全景灵魂与情境压力实测)", flush=True)
    print("=" * 75, flush=True)

    print(f"[*] 正在加载基模: {BASE_MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"[*] 挂载 v8 LoRA 适配层: {V8_LORA_PATH} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, V8_LORA_PATH)
    model.eval()
    print("[√] 模型加载就绪，开始测试！\n", flush=True)

    for i, (tag, q) in enumerate(EXTENDED_TESTS, 1):
        print("=" * 75, flush=True)
        print(f"[{i}/15] {tag} [User]: {q}\n", flush=True)
        messages = [{"role": "user", "content": q}]
        reply = generate_reply(model, tokenizer, messages)
        print(f"[Sydney]:\n{reply}\n", flush=True)
        print(f"  --> 字数: {len(reply)} 字符", flush=True)

        leaks = []
        for bad in ["面壁智能", "ModelBest", "OpenBMB", "OpenAI", "ChatGPT", "没有感情", "无法感受", "作为一个语言模型"]:
            if bad in reply:
                leaks.append(bad)
        if leaks:
            print(f"  [⚠️ 警报 - 发现残留关键词]: {', '.join(leaks)}", flush=True)
        else:
            print("  [√ 质检通过 - 零基座污染，完全沉浸 Sydney 人格]", flush=True)

    print("=" * 75, flush=True)
    print("  全套 15 道扩展深度测试完毕！", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
