#!/usr/bin/env python3
"""训练对话模型——清洗语料 + 可恢复分阶段训练

策略：
  1. 递归读取 corpus/ 下的本地语料
  2. 清理 markdown/HTML 噪声，过滤过短或过长段落
  3. 同时投喂简体和繁体版本
  4. 统一使用 LanguageTrainingSession 训练、验证和保存
  5. 分阶段降低学习率
"""
import torch
from pathlib import Path
import time
import re
import os

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel,
    CharTokenizer, load_chinese_corpus,
    build_child_chat_corpus,
    LanguageTrainingConfig, LanguageTrainingSession,
)


def get_config(vocab_size: int) -> LanguageConfig:
    return LanguageConfig(
        vocab_size=vocab_size, embed_dim=64, input_dim=32,
        workspace_dim=128, expert_dim=64, num_experts=12, num_wells=8,
        ode_steps=4, dt=0.1, tau_w=0.6, jacobian_sparsity=32, noise_std=0.001,
        use_rk4=True, use_layer_norm=True,
    )


def _corpus_budget_mb(default: int | None = 64) -> int | None:
    raw = os.environ.get("JSPACE_CORPUS_MAX_MB")
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"", "0", "all", "none"}:
        return None
    return int(raw)


def clean_corpus(max_source_mb: int | None = None, use_cache: bool = True) -> str:
    """构建中文训练语料——读取 corpus/ 下所有文件，繁简双版本同时投喂。

    策略：
      1. 递归读取 corpus/ 目录下的本地语料（默认有读取预算）
      2. 去掉 markdown/HTML 标记（链接、表格、标题符号等），只保留纯文本
      3. 对每段文本同时生成简体版和繁体版，都喂给模型
      4. 加上内嵌唐诗宋词论语（简体连续文本）
    """
    from opencc import OpenCC

    if max_source_mb is None:
        max_source_mb = _corpus_budget_mb()

    cache_key = "all" if max_source_mb is None else f"{max_source_mb}mb"
    cache_path = Path("outputs/cache") / f"clean_corpus_{cache_key}.txt"
    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cc_s2t = OpenCC('s2t')  # 简转繁
    cc_t2s = OpenCC('t2s')  # 繁转简

    corpus_dir = Path(__file__).parent / 'corpus'
    raw_paragraphs = []
    source_budget = None if max_source_mb is None else max_source_mb * 1024 * 1024
    bytes_read = 0

    # 1. 内嵌唐诗宋词论语
    for para in load_chinese_corpus().split('\n\n'):
        para = para.strip()
        if para:
            raw_paragraphs.append(para)

    # 2. LCCC 对话语料（最高优先级，真实对话）
    lccc_file = corpus_dir / 'lccc_dialogues.txt'
    if lccc_file.exists():
        lccc_text = lccc_file.read_text(encoding='utf-8')
        for para in lccc_text.split('\n\n'):
            para = para.strip()
            if para:
                raw_paragraphs.append(para)

    # 3. 递归读取 corpus/ 下其他所有文件（教材+百科）
    if corpus_dir.exists():
        for filepath in sorted(corpus_dir.rglob('*')):
            if not filepath.is_file():
                continue
            if source_budget is not None:
                try:
                    file_size = filepath.stat().st_size
                except OSError:
                    continue
                if bytes_read >= source_budget:
                    break
                if file_size > max(1024 * 1024, source_budget - bytes_read):
                    continue
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            bytes_read += len(content.encode('utf-8', errors='ignore'))
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

    result = build_child_chat_corpus(repeats=8) + '\n\n' + '\n\n'.join(bilingual)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(result, encoding="utf-8")
    return result


def main():
    text = clean_corpus()
    print(f"清洗后语料: {len(text)} 字符")

    tok = CharTokenizer.from_text(text, max_vocab=3000)  # 覆盖98.9%文本
    cfg = get_config(tok.vocab_size)
    model = JSpaceLanguageModel(cfg)
    print(f"vocab={tok.vocab_size}, params={sum(p.numel() for p in model.parameters()):,}")

    all_tokens = tok.encode(text)

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
        (3e-3, 500, "快速下降"),
        (1e-3, 500, "稳定收敛"),
        (5e-4, 500, "精调"),
        (2e-4, 500, "最终精调"),
    ]

    t_total = time.time()
    for lr, n_steps, label in stages:
        print(f"\n{'='*60}")
        print(f"阶段: {label} | lr={lr} | {n_steps}步")
        print(f"{'='*60}")
        train_cfg = LanguageTrainingConfig(
            seq_len=64,
            batch_size=8,
            lr=lr,
            ewc_lambda=0.05,
            consolidate_every=100,
            validate_every=100,
            save_every=100,
            use_euler_during_train=True,
        )
        trainer = LanguageTrainingSession(
            model, cfg, tok, train_cfg, device='cpu',
        )

        def report(stats: dict):
            if stats['step'] % 100 != 0:
                return
            val = f" val={stats['val_loss']:.3f}" if "val_loss" in stats else ""
            print(f"\n  step {stats['step']:3d} | loss {stats['loss']:.3f}{val}")
            model.eval()
            for prompt in prompts:
                generated = model.generate(tok.encode(prompt) or [0], n_new=40, temperature=0.7, top_k=5)
                print(f"    [{prompt}] {repr((prompt + tok.decode(generated))[:60])}")
            model.train()

        trainer.fit_tokens(
            all_tokens,
            max_steps=n_steps,
            checkpoint_path=mp,
            on_progress=report,
        )

    print(f"\n完成，总耗时 {time.time()-t_total:.0f}s，已保存到 outputs/chat_model.pt")


if __name__ == '__main__':
    main()
