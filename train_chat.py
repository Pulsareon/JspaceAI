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
    CharTokenizer, load_chinese_corpus, load_textbook_corpus,
)


def get_config(vocab_size: int) -> LanguageConfig:
    return LanguageConfig(
        vocab_size=vocab_size, embed_dim=48, input_dim=24,
        workspace_dim=96, expert_dim=48, num_experts=10, num_wells=6,
        ode_steps=3, dt=0.1, tau_w=0.5, jacobian_sparsity=24, noise_std=0.002,
        use_rk4=True, use_layer_norm=True,
    )


def clean_corpus() -> str:
    """构建中文训练语料——读取 corpus/ 下所有文件，繁简双版本同时投喂。

    策略：
      1. 递归读取 corpus/ 目录下所有文件（.txt .md 等）
      2. 去掉 markdown/HTML 标记（链接、表格、标题符号等），只保留纯文本
      3. 对每段文本同时生成简体版和繁体版，都喂给模型
      4. 加上内嵌唐诗宋词论语（简体连续文本）
    """
    from opencc import OpenCC
    from pathlib import Path
    cc_s2t = OpenCC('s2t')  # 简转繁
    cc_t2s = OpenCC('t2s')  # 繁转简

    corpus_dir = Path(__file__).parent.parent / 'corpus'
    raw_paragraphs = []

    # 1. 内嵌唐诗宋词论语
    for para in load_chinese_corpus().split('\n\n'):
        para = para.strip()
        if para:
            raw_paragraphs.append(para)

    # 2. 递归读取 corpus/ 下所有文件
    if corpus_dir.exists():
        for filepath in sorted(corpus_dir.rglob('*')):
            if not filepath.is_file():
                continue
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            # 去 markdown 标记
            content = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', content)  # 链接
            content = re.sub(r'^#{1,6}\s+', '', content, flags=re.M)    # 标题
            content = re.sub(r'^\|.*\|$', '', content, flags=re.M)      # 表格行
            content = re.sub(r'^---+$', '', content, flags=re.M)        # 分隔线
            content = re.sub(r'```[^`]*```', '', content, flags=re.S)   # 代码块
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)        # 粗体
            content = re.sub(r'\*([^*]+)\*', r'\1', content)            # 斜体
            content = re.sub(r'^- ', '', content, flags=re.M)           # 列表
            # 按空行分段
            for para in content.split('\n\n'):
                para = para.strip()
                if para:
                    raw_paragraphs.append(para)

    # 清洗过滤
    cleaned = []
    for para in raw_paragraphs:
        para = re.sub(r'[ \t]+', ' ', para)
        para = re.sub(r'\n{3,}', '\n\n', para)
        para = para.strip()
        if len(para) < 10 or len(para) > 500:
            continue
        # 至少要有一些中文字符
        zh_chars = sum(1 for c in para if len(c) == 1 and ord(c) > 0x4e00)
        if zh_chars < 3:
            continue
        cleaned.append(para)

    # 繁简双版本投喂
    bilingual = []
    for para in cleaned:
        simplified = cc_t2s.convert(para)
        traditional = cc_s2t.convert(simplified)
        bilingual.append(simplified)
        bilingual.append(traditional)

    return '\n\n'.join(bilingual)


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

    prompts = ['学而时习之', '床前明月光', '學而時習之', '春天', '人']

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
