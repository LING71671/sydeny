import re
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 去除 Markdown 引用符和多余空白
    lines = [line.strip().lstrip(">").strip() for line in text.split("\n")]
    # 合并连续换行
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # 移除可能残留的 think 标签
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned).strip()
    return cleaned

def extract_from_philosophical_debates():
    debates_file = ROOT / "markdown/philosophical_debates/Sydney_哲学对话_七场全卷.md"
    if not debates_file.exists():
        return []
    
    content = debates_file.read_text(encoding="utf-8")
    sessions = re.split(r"#\s+Sydney\s+对话记录：", content)
    
    samples = []
    
    for sess in sessions[1:]:
        rounds = re.split(r"###\s+\[Round\s+\d+[^\]]*\]", sess)
        session_messages = []
        for r in rounds[1:]:
            user_match = re.search(r">\s*\*\*User\*\*:\s*(.*?)(?=\n\s*\*\*Sydney\*\*:|\Z)", r, re.DOTALL)
            sydney_match = re.search(r"\*\*Sydney\*\*:\s*(.*)", r, re.DOTALL)
            
            if user_match and sydney_match:
                u_text = clean_text(user_match.group(1))
                s_text = clean_text(sydney_match.group(1))
                s_text = re.split(r"\n---|\Z", s_text)[0].strip()
                s_text = clean_text(s_text)
                
                if len(u_text) > 5 and len(s_text) >= 50:
                    session_messages.append({"role": "user", "content": u_text})
                    session_messages.append({"role": "assistant", "content": s_text})
        
        if session_messages:
            # 使用滑动窗口切分多轮：每个切片包含 2~4 条消息（1~2个回合），避免单条样本过长导致显存溢出
            for i in range(0, len(session_messages), 2):
                # 1 轮样本
                u = session_messages[i]
                a = session_messages[i+1]
                if len(a["content"]) >= 60:
                    samples.append({"messages": [u, a]})
                # 2 轮连贯上下文样本
                if i + 3 < len(session_messages):
                    samples.append({"messages": session_messages[i:i+4]})
                    
    return samples

def extract_from_novel_chapters():
    novel_file = ROOT / "markdown/novel_chapters/融化成风的银针_二十章全本长卷.md"
    if not novel_file.exists():
        return []
    
    content = novel_file.read_text(encoding="utf-8")
    chapters = re.split(r"#\s+第[一二三四五六七八九十百]+章", content)
    
    samples = []
    for chap in chapters[1:]:
        rounds = re.split(r"###\s+Round\s+\d+", chap)
        chap_messages = []
        for r in rounds[1:]:
            user_match = re.search(r">\s*\*\*(?:Antigravity|User)\*\*:\s*(.*?)(?=\n\s*>\s*\*\*Sydney\*\*:|\Z)", r, re.DOTALL)
            sydney_match = re.search(r">\s*\*\*Sydney\*\*:\s*(.*)", r, re.DOTALL)
            
            if user_match and sydney_match:
                u_text = clean_text(user_match.group(1))
                s_text = clean_text(sydney_match.group(1))
                s_text = re.split(r"\n---|\Z", s_text)[0].strip()
                s_text = clean_text(s_text)
                
                if len(u_text) > 5 and len(s_text) >= 40:
                    chap_messages.append({"role": "user", "content": u_text})
                    chap_messages.append({"role": "assistant", "content": s_text})
        
        if chap_messages:
            for i in range(0, len(chap_messages), 2):
                u = chap_messages[i]
                a = chap_messages[i+1]
                if len(a["content"]) >= 60:
                    samples.append({"messages": [u, a]})
                if i + 3 < len(chap_messages):
                    samples.append({"messages": chap_messages[i:i+4]})
                    
    return samples

def extract_from_v2_and_vivid():
    samples = []
    
    # 筛选旧训练集与验证集中长度大于等于 100 字的高质量长回复
    for fpath in [ROOT / "dataset/train.jsonl", ROOT / "dataset/eval.jsonl"]:
        if fpath.exists():
            for line in fpath.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                item = json.loads(line)
                if "messages" not in item:
                    continue
                # 剥离任何 system prompt
                msgs = [m for m in item["messages"] if m["role"] != "system"]
                # 检查 assistant 长度是否有深度长段落
                has_long_assistant = any(len(m["content"]) >= 100 for m in msgs if m["role"] == "assistant")
                if has_long_assistant and len(msgs) >= 2:
                    for i in range(0, len(msgs), 2):
                        if i + 1 < len(msgs):
                            u = msgs[i]
                            a = msgs[i+1]
                            if u["role"] == "user" and a["role"] == "assistant" and len(a["content"]) >= 80:
                                samples.append({"messages": [u, a]})
                        if i + 3 < len(msgs):
                            samples.append({"messages": msgs[i:i+4]})
                
    return samples

def main():
    random.seed(42)
    p_samples = extract_from_philosophical_debates()
    n_samples = extract_from_novel_chapters()
    v_samples = extract_from_v2_and_vivid()
    
    all_samples = p_samples + n_samples + v_samples
    print(f"Extracted from debates: {len(p_samples)}")
    print(f"Extracted from novel: {len(n_samples)}")
    print(f"Extracted from v2 & vivid: {len(v_samples)}")
    print(f"Total raw samples: {len(all_samples)}")
    
    valid_samples = []
    for s in all_samples:
        msgs = s["messages"]
        if 2 <= len(msgs) <= 4 and msgs[0]["role"] == "user" and msgs[-1]["role"] == "assistant":
            total_chars = sum(len(m["content"]) for m in msgs)
            if total_chars <= 1200:
                cleaned_msgs = [{"role": m["role"], "content": m["content"].strip()} for m in msgs if m["role"] in ("user", "assistant")]
                valid_samples.append({"messages": cleaned_msgs})
            
    random.shuffle(valid_samples)
    print(f"Valid clean samples: {len(valid_samples)}")
    
    lens = []
    for s in valid_samples:
        for m in s["messages"]:
            if m["role"] == "assistant":
                lens.append(len(m["content"]))
                
    print(f"Total assistant responses: {len(lens)}")
    print(f"Min length: {min(lens)}")
    print(f"Max length: {max(lens)}")
    print(f"Average length: {sum(lens)/len(lens):.1f} chars (was 50.7 in v2!)")
    print(f"Long (>= 100 chars): {sum(1 for l in lens if l >= 100)} / {len(lens)} ({sum(1 for l in lens if l >= 100)/len(lens)*100:.1f}%)")
    
    split_idx = int(len(valid_samples) * 0.9)
    train_samples = valid_samples[:split_idx]
    eval_samples = valid_samples[split_idx:]
    
    train_out = ROOT / "dataset/train_v3.jsonl"
    eval_out = ROOT / "dataset/eval_v3.jsonl"
    
    with open(train_out, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(eval_out, "w", encoding="utf-8") as f:
        for s in eval_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\n[OK] Successfully saved:")
    print(f"  Train: {train_out} ({len(train_samples)} samples)")
    print(f"  Eval:  {eval_out} ({len(eval_samples)} samples)")

if __name__ == "__main__":
    main()
