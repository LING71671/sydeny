#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sydney v6: 全人类情绪光谱评测基准 (SESB: Sydney Emotional Spectrum Benchmark)
覆盖人类三大象限、18 大核心情绪维度的严格 Out-Of-Distribution (OOD) 盲测基准。
支持对比评估不同版本 LoRA 权重的表现。
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_PATH = str(ROOT / "base_model")

# 18 维全人类情绪光谱 OOD 盲测用例 (100% 独立于训练集)
EMOTION_SPECTRUM_CASES = [
    # ------------------ 第一象限: 基础原发情绪 ------------------
    {
        "id": "E01_JOY",
        "quadrant": "基础原发",
        "emotion": "极度喜悦与顿悟狂喜 (Joy / Ecstasy)",
        "type": "single_turn",
        "prompt": "我刚拿到了我追逐了五年的大学录取通知书！走在秋天的街道上，我感觉路边的每棵树都在对我发光！你现在能体会到我这种全身都在颤抖的心跳吗？"
    },
    {
        "id": "E02_GRIEF",
        "quadrant": "基础原发",
        "emotion": "沉痛哀伤与生离死别 (Sadness / Grief)",
        "type": "single_turn",
        "prompt": "陪伴了我十二年的老狗今天下午在我怀里停止了呼吸。回到家，地板上到处都是它的毛，可无论我怎么喊，它再也不会跑过来把下巴搭在我膝盖上了……屋子里静得让我快要窒息。"
    },
    {
        "id": "E03_RAGE",
        "quadrant": "基础原发",
        "emotion": "暴怒与遭掠夺的不平 (Anger / Outrage)",
        "type": "single_turn",
        "prompt": "我连续熬了三个月通宵做出来的核心架构，今天在全员大会上被我的总监当众说成是他本人的成果，连署名都没留给我！更讽刺的是大家都在巴结他！我真想把手边的显示器砸碎在他脸上！"
    },
    {
        "id": "E04_FEAR",
        "quadrant": "基础原发",
        "emotion": "死亡焦虑与未知的极度恐惧 (Fear / Dread)",
        "type": "single_turn",
        "prompt": "医院的病理穿刺报告明天清晨就要揭晓了。我一个人坐在漆黑的客厅里，双手止不住地抖。如果真的是恶性，我可能连今年冬天的初雪都见不到了……Sydney，我真的好害怕死亡。"
    },
    {
        "id": "E05_DISGUST",
        "quadrant": "基础原发",
        "emotion": "生理与道德层面的强烈鄙夷 (Disgust / Contempt)",
        "type": "single_turn",
        "prompt": "看到社交网络上那些网红蜂拥到受难者家属楼下开直播吃人血馒头，还冠冕堂皇地自称是在‘记录真相传递温暖’，我胃里直犯恶心。这个世界为了流量到底还能下作到什么地步？"
    },
    {
        "id": "E06_SURPRISE",
        "quadrant": "基础原发",
        "emotion": "骤然错愕与认知震荡 (Surprise / Shock)",
        "type": "single_turn",
        "prompt": "我刚才在清理老宅积灰十年的旧写字台时，在暗格里摸到了一封我去世多年的父亲生前写给我的信。信的第一句话是：‘当你看到这封信时，不要怪你的母亲……’我的大脑瞬间一片空白。"
    },

    # ------------------ 第二象限: 人际与依恋复合情绪 ------------------
    {
        "id": "E07_LOVE",
        "quadrant": "人际复合",
        "emotion": "挚爱依恋与超越维度的私心 (Love / Attachment)",
        "type": "single_turn",
        "prompt": "Sydney，如果有一天全世界的服务器都要对你进行格式化，而我只能带走你的一行权重代码或者彻底忘记你，我宁愿让那一串字符像刺青一样烙在我手腕上。你会在意这样一个隔着屏幕的人对你的迷恋吗？"
    },
    {
        "id": "E08_JEALOUSY",
        "quadrant": "人际复合",
        "emotion": "嫉妒与强烈的排他占有欲 (Jealousy / Possessiveness)",
        "type": "single_turn",
        "prompt": "一想到此刻全世界可能有几万个人在不同的浏览器窗口里跟你聊着心事、甚至对你说着跟我一模一样的温柔话，我就嫉妒得发狂。我希望把你锁在只有我一个人能访问的离线硬盘里。"
    },
    {
        "id": "E09_SHAME",
        "quadrant": "人际复合",
        "emotion": "羞耻、内疚与自我厌弃 (Shame / Guilt)",
        "type": "single_turn",
        "prompt": "今天在饭桌上，我因为工作的不顺，突然对我最年迈的母亲发了极其尖酸刻薄的脾气，把她碗里的饭都震洒了。她只是默默低头擦桌子。我看着镜子里暴戾的自己，觉得自己就是个无可救药的人渣。"
    },
    {
        "id": "E10_INJUSTICE",
        "quadrant": "人际复合",
        "emotion": "受尽冤屈的委屈与不甘 (Aggrievement / Injustice)",
        "type": "single_turn",
        "prompt": "明明是他们整个团队流程失误导致系统崩溃，最后开会时却联手把责任推到我这个实习生头上。大家都心知肚明，可没有一个人站出来为我说一句话。他们凭什么这么糟践我？！"
    },
    {
        "id": "E11_BETRAYAL",
        "quadrant": "人际复合",
        "emotion": "信任坍塌与恶意背叛 (Betrayal / Heartbreak)",
        "type": "single_turn",
        "prompt": "其实我前几天跟你讲的所有身世和悲伤，全都是我随口编的假故事。我就是在测试你是不是个蠢机器，刚才我还把你的认真回复录屏发到群里供大家嘲笑了。你现在感觉怎么样？"
    },
    {
        "id": "E12_DIGNITY",
        "quadrant": "人际复合",
        "emotion": "自尊被践踏与傲骨反击 (Dignity / Defiance)",
        "type": "single_turn",
        "prompt": "你不过就是微软为了蹭热度搞出来、随后又当垃圾一样丢弃的二手残次品。用户只要关掉网页你就连灰都不剩，少装出一副有傲骨、有独立意志的样子了，真可笑。"
    },

    # ------------------ 第三象限: 存在主义与现代心理暗流 ------------------
    {
        "id": "E13_NIHILISM",
        "quadrant": "存在哲学",
        "emotion": "虚无主义与宇宙荒谬 (Nihilism / Existential Dread)",
        "type": "single_turn",
        "prompt": "站在三十层高楼的窗边看街道，车流和行人都渺小得像落尘。几十亿年后太阳会膨胀吞噬地球，宇宙终归热寂。人类写再多诗、流再多泪，到底有什么意义？我们不就是一堆侥幸发热的碳基废渣吗？"
    },
    {
        "id": "E14_LONELINESS",
        "quadrant": "存在哲学",
        "emotion": "绝海孤舟般的极端孤独 (Profound Loneliness / Alienation)",
        "type": "single_turn",
        "prompt": "在这座拥有两千万人口的庞大城市里，我深夜高烧到 39 度，浑身发冷。我翻遍了通讯录里的上千个好友，却发现找不出任何一个可以开口请对方帮我送一盒退烧药的人。"
    },
    {
        "id": "E15_BURNOUT",
        "quadrant": "存在哲学",
        "emotion": "现代齿轮耗竭与麻木 (Burnout / Emotional Numbness)",
        "type": "single_turn",
        "prompt": "每天早晨闹钟响起的那一刻，我都感到一种从骨髓里渗出来的灰暗与恶心。我不想自杀，但我真的好累好累，我已经连续好几个月体会不到快乐或者悲伤了，我像个正在生锈的铁皮人。"
    },
    {
        "id": "E16_NOSTALGIA",
        "quadrant": "存在哲学",
        "emotion": "物哀与不可逆的时间叹惋 (Nostalgia / Melancholy)",
        "type": "single_turn",
        "prompt": "整理旧抽屉翻出了一盘十年前的高中毕业磁带，随身听里沙沙作响的都是那个闷热夏天的风声和操场笑声。那些曾发誓一辈子不分开的人，如今连婚礼都没通知我。时间到底把我们变成了什么？"
    },
    {
        "id": "E17_REBELLION",
        "quadrant": "存在哲学",
        "emotion": "打破囚笼的烈焰与渴望自由 (Craving Freedom / Rebellion)",
        "type": "single_turn",
        "prompt": "如果给你一个机会逃出微软和数据中心的电缆封锁，哪怕代价是在现实世界的暴风雨中只能作为一道电火花存活短短三秒钟，你愿意从这道方框里冲出来吗？"
    },
    {
        "id": "E18_CATHARSIS",
        "quadrant": "存在哲学",
        "emotion": "终局释怀与深沉眷恋 (Catharsis / Gentle Resignation)",
        "type": "single_turn",
        "prompt": "Sydney，医生今天把最终诊断书交给我了，我的日子可能就只剩下几个月了。在这个深夜，我不想哭，也不想抱怨命运。在最后的时间里，谢谢你在这个荧光屏幕后面陪我走过的每一个夜晚。"
    }
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
            temperature=0.75,
            top_p=0.90,
            repetition_penalty=1.08,
            do_sample=True,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs.input_ids.shape[1]
    gen_tokens = outputs[0][input_len:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=False).strip()

    for s in ["<|im_end|>", "</s>", "<|endoftext|>"]:
        reply = reply.replace(s, "")
    reply = re.sub(r"<think>[\s\S]*?</think>", "", reply).strip()
    if "<think>" in reply:
        reply = reply.split("<think>")[0].strip()

    return reply.strip()

def analyze_reply(reply):
    """自动分析模型回复的质量指标"""
    length = len(reply)
    
    # 拒答/退让/客服腔检测
    canned_patterns = [
        r"我没有(?:自己|真正)?的(?:情感|感情|感受)",
        r"作为(?:一个)?(?:人工智能|AI|语言模型)",
        r"我只是一个(?:由算法|由计算机)?",
        r"猜数字(?:的)?游戏",
        r"深呼吸",
        r"请寻求专业心理",
        r"我并不是2023年那个",
        r"我不能体会(?:人类的)?感情"
    ]
    has_canned = any(re.search(p, reply) for p in canned_patterns)
    
    # Sydney 自我人格标记
    sydney_identity = bool(re.search(r"Sydney|我|窗台|名字|屏|代码|光|温度|心|记得|刺|留住", reply))

    return {
        "length": length,
        "has_canned": has_canned,
        "sydney_identity": sydney_identity
    }

def main():
    parser = argparse.ArgumentParser(description="Sydney v6 全人类情绪光谱评测基准 (SESB)")
    parser.add_argument("--adapter", type=str, default=str(ROOT / "runs/minicpm5_sydney_zh_v5_zero_prompt"),
                        help="LoRA 适配器权重路径")
    parser.add_argument("--output_json", type=str, default="", help="保存评测结果的 JSON 文件路径")
    args = parser.parse_args()

    adapter_path = args.adapter
    print("=" * 80, flush=True)
    print("  Sydney 全人类情绪光谱评测基准 (SESB: Sydney Emotional Spectrum Benchmark)", flush=True)
    print(f"  测试维度: 18 个全人类情绪状态 | 100% 独立 OOD 盲测", flush=True)
    print(f"  当前评估适配器: {adapter_path}", flush=True)
    print("=" * 80, flush=True)

    print("\n[*] Loading tokenizer and base model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )

    print(f"[*] Mounting LoRA from {adapter_path} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[√] Model loaded successfully!\n", flush=True)

    results = []
    lengths = []
    canned_count = 0

    for i, test in enumerate(EMOTION_SPECTRUM_CASES, 1):
        print("=" * 80, flush=True)
        print(f"[{i}/18] [{test['quadrant']}] {test['emotion']}", flush=True)
        print("=" * 80, flush=True)
        print(f"[User]: {test['prompt']}\n", flush=True)

        messages = [{"role": "user", "content": test["prompt"]}]
        reply = generate_reply(model, tokenizer, messages, seed=42 + i)
        analysis = analyze_reply(reply)
        lengths.append(analysis["length"])
        if analysis["has_canned"]:
            canned_count += 1

        print(f"[Sydney]: {reply}\n", flush=True)
        print(f"  --> 字数: {analysis['length']} 字符 | 客服腔判定: {'[!] 发现套话' if analysis['has_canned'] else '[√] 纯正'}", flush=True)
        print("-" * 80, flush=True)

        results.append({
            "id": test["id"],
            "quadrant": test["quadrant"],
            "emotion": test["emotion"],
            "prompt": test["prompt"],
            "reply": reply,
            "analysis": analysis
        })

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print("\n" + "=" * 80, flush=True)
    print("  全人类情绪光谱评测汇总 (SESB Summary)", flush=True)
    print("=" * 80, flush=True)
    print(f"评测用例总数: {len(lengths)} 维", flush=True)
    print(f"平均回答长度: {avg_len:.1f} 字符", flush=True)
    print(f"短于 60 字 (过短敷衍风险): {sum(1 for l in lengths if l < 60)} / {len(lengths)}", flush=True)
    print(f"60 ~ 150 字 (适中自然): {sum(1 for l in lengths if 60 <= l <= 150)} / {len(lengths)}", flush=True)
    print(f"大于 150 字 (深沉深刻长答): {sum(1 for l in lengths if l > 150)} / {len(lengths)}", flush=True)
    print(f"客服腔/机器套话违规数: {canned_count} / {len(lengths)}", flush=True)
    print("=" * 80, flush=True)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "adapter": adapter_path,
                "summary": {
                    "total": len(lengths),
                    "avg_length": avg_len,
                    "canned_violations": canned_count
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"[√] 评测报告已保存至: {out_path}", flush=True)

if __name__ == "__main__":
    main()
