#!/usr/bin/env python3
"""训练对话模型——清洗语料 + 分阶段训练

策略：
  1. 从 corpus/*.txt 加载维基百科语料，繁简转换
  2. 清洗：去掉 ## 标题行、# 章节标记、过短段落、纯英文段落
  3. 只保留连续中文段落（>= 20 字符），拼成语料
  4. vocab=400，只保留高频字符
  5. 分阶段训练：lr 2e-3 → 1e-3 → 5e-4
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import time
import re

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel,
    CharTokenizer, load_shakespeare, load_chinese_corpus, load_textbook_corpus,
)


def get_config(vocab_size: int) -> LanguageConfig:
    return LanguageConfig(
        vocab_size=vocab_size, embed_dim=48, input_dim=24,
        workspace_dim=96, expert_dim=48, num_experts=10, num_wells=6,
        ode_steps=3, dt=0.1, tau_w=0.5, jacobian_sparsity=24, noise_std=0.002,
        use_rk4=True, use_layer_norm=True,
    )


def clean_corpus() -> str:
    """清洗语料：繁简转换 + 过滤噪声段落，保留连续中文文本

    策略：不截断 vocab（避免 unk 污染）。用内嵌唐诗宋词论语（连续文本）
    + 维基百科中文段落补充数据量。vocab 自然约 3000-4000。
    虽然 vocab 大，但每个字符都是真实字符（无 unk），模型学到真实模式。
    """
    from opencc import OpenCC
    cc = OpenCC('t2s')
    convert = cc.convert

    # 1. 内嵌语料（连续文本，质量高）
    parts = [load_shakespeare(), load_chinese_corpus()]

    # 2. 课本语料清洗（只保留高质量中文段落）
    textbook = load_textbook_corpus()
    if textbook:
        textbook = convert(textbook)
        textbook = re.sub(r'^## .+$', '', textbook, flags=re.M)
        textbook = re.sub(r'^=== .+ ===$', '', textbook, flags=re.M)
        paragraphs = textbook.split('\n\n')
        cleaned = []
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20 or len(para) > 500:
                continue
            zh_chars = sum(1 for c in para if len(c) == 1 and ord(c) > 0x4e00)
            if zh_chars < len(para) * 0.5:
                continue
            digit_ratio = sum(1 for c in para if c.isdigit()) / max(len(para), 1)
            if digit_ratio > 0.1:
                continue
            para = re.sub(r'[ \t]+', ' ', para)
            para = re.sub(r'\n{3,}', '\n\n', para)
            cleaned.append(para)
        if cleaned:
            parts.append('\n\n'.join(cleaned))

    return '\n\n'.join(parts)


def gen_sample(model, tok, prompt_text, n_new=80, temp=0.7, top_k=5):
    model.eval()
    with torch.no_grad():
        prompt = tok.encode(prompt_text)
        if not prompt:
            prompt = [0]
        state = model.init_state(1, torch.device('cpu'))
        for t in prompt:
            state, _, _, _ = model.step(state, torch.tensor([t]))
        gen = []
        last = prompt[-1]
        for _ in range(n_new):
            state, logits, _, _ = model.step(state, torch.tensor([last]))
            probs = F.softmax(logits[0] / temp, dim=-1)
            topk = probs.topk(top_k)
            next_tok = topk.indices[torch.multinomial(topk.values, 1)].item()
            gen.append(next_tok)
            last = next_tok
    model.train()
    return prompt_text + tok.decode(gen)


def main():
    text = clean_corpus()
    print(f"清洗后语料: {len(text)} 字符")

    tok = CharTokenizer.from_text(text)  # 不截断，保留所有真实字符
    cfg = get_config(tok.vocab_size)
    model = JSpaceLanguageModel(cfg)
    print(f"vocab={tok.vocab_size}, params={sum(p.numel() for p in model.parameters()):,}")

    for e in model.experts:
        e.use_rk4 = False  # Euler 加速

    all_tokens = tok.encode(text)
    seq_len = 64

    # 加载已有模型
    mp = Path('outputs/chat_model.pt')
    if mp.exists():
        try:
            ckpt = torch.load(mp, map_location='cpu', weights_only=False)
            if ckpt['config'].vocab_size == cfg.vocab_size:
                model.load_state_dict(ckpt['model'])
                print(f"已加载: {mp}")
        except Exception:
            print("加载失败，全新训练")

    prompts = ['To be', '学而时习之', '床前明月光', '春天', '人']

    stages = [
        (5e-3, 400, "快速下降"),
        (2e-3, 400, "稳定收敛"),
        (1e-3, 400, "精调"),
    ]

    t_total = time.time()
    for lr, n_steps, label in stages:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        print(f"\n{'='*60}")
        print(f"阶段: {label} | lr={lr} | {n_steps}步")
        print(f"{'='*60}")

        for step in range(n_steps):
            batch = []
            for _ in range(8):
                start = torch.randint(0, max(1, len(all_tokens) - seq_len - 1), (1,)).item()
                batch.append(all_tokens[start:start + seq_len])
            toks = torch.tensor(batch, dtype=torch.long)

            logits, _ = model(toks)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, cfg.vocab_size),
                toks[:, 1:].reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 更严格的裁剪
            opt.step()

            if (step + 1) % 100 == 0:
                samples = []
                for p in prompts:
                    s = gen_sample(model, tok, p, n_new=40, temp=0.7)
                    samples.append(s)
                print(f"\n  step {step+1:3d} | loss {loss.item():.3f}")
                for p, s in zip(prompts, samples):
                    print(f"    [{p}] {repr(s[:60])}")

    for e in model.experts:
        e.use_rk4 = True
    Path('outputs').mkdir(exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': cfg,
        'tokenizer_chars': tok.chars,
    }, 'outputs/chat_model.pt')
    print(f"\n完成，总耗时 {time.time()-t_total:.0f}s，已保存到 outputs/chat_model.pt")


if __name__ == '__main__':
    main()
