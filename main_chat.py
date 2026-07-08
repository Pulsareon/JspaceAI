#!/usr/bin/env python3
"""
JspaceAI —— 对话版（只有语言，控制台交互）

基于字符级语言模型 + 自主进化（EWC + 经验回放 + 专家可塑性）。
交互式对话：用户输入文本，模型生成回复，同时持续学习。

模式：
    --mode chat:     交互对话（默认）
    --mode train:    在清洗后的中文语料上训练若干步
    --mode generate: 给定提示词一次性生成

运行：
    python main_chat.py
    python main_chat.py --mode train --steps 100
    python main_chat.py --mode generate --prompt "To be"
"""
from __future__ import annotations
import argparse
import torch
from pathlib import Path

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel,
    OnlineLanguageLearner,
    CharTokenizer,
    LanguageTrainingConfig, LanguageTrainingSession,
    expert_integration_mode, save_language_checkpoint,
)
from train_chat import clean_corpus


def tokenizer_from_chars(chars: list[str]) -> CharTokenizer:
    return CharTokenizer(
        chars=chars,
        char_to_idx={c: i for i, c in enumerate(chars)},
        idx_to_char={i: c for i, c in enumerate(chars)},
    )


def get_config(vocab_size: int) -> LanguageConfig:
    return LanguageConfig(
        vocab_size=vocab_size,
        embed_dim=64, input_dim=32,
        workspace_dim=128, expert_dim=64,
        num_experts=12, num_wells=8,
        ode_steps=4, dt=0.1, tau_w=0.6,
        jacobian_sparsity=32, noise_std=0.001,
        use_rk4=True, use_layer_norm=True,
    )


def load_or_init_model(device: str):
    """加载已保存的模型或初始化新模型"""
    mp = Path('outputs/chat_model.pt')
    if mp.exists():
        try:
            ckpt = torch.load(mp, map_location=device, weights_only=False)
            # 用保存的 tokenizer chars 确保一致
            saved_chars = ckpt.get('tokenizer_chars', None)
            if saved_chars:
                tokenizer = tokenizer_from_chars(saved_chars)
                text = None
            else:
                print("checkpoint 缺少 tokenizer，正在准备语料...")
                text = clean_corpus()
                tokenizer = CharTokenizer.from_text(text)
            config = get_config(tokenizer.vocab_size)
            model = JSpaceLanguageModel(config).to(device)
            model.load_state_dict(ckpt['model'])
            print(f"已加载模型: {mp}（vocab={tokenizer.vocab_size}）")
            return model, config, tokenizer, text
        except Exception as e:
            print(f"模型加载失败: {e}，全新初始化")

    text = clean_corpus()  # 清洗后语料（繁简统一+过滤）
    tokenizer = CharTokenizer.from_text(text)
    config = get_config(tokenizer.vocab_size)
    model = JSpaceLanguageModel(config).to(device)
    print("全新初始化（首次对话）")
    return model, config, tokenizer, text


def save_model(model, config, tokenizer):
    save_language_checkpoint('outputs/chat_model.pt', model, config, tokenizer)


def ensure_corpus(text: str | None) -> str:
    if text is None:
        print("正在准备训练语料...")
        return clean_corpus()
    return text


def train(model, tokenizer, text: str | None, n_steps: int, device: str):
    """Scalable language-model training entrypoint."""
    print("\n" + "=" * 60)
    print(f"训练 {n_steps} 步")
    print("=" * 60)

    text = ensure_corpus(text)
    config = model.config
    train_cfg = LanguageTrainingConfig(
        seq_len=64,
        batch_size=8,
        lr=5e-3,
        ewc_lambda=0.05,
        consolidate_every=50,
        validate_every=max(1, min(50, n_steps)),
        save_every=max(1, min(100, n_steps)),
        use_euler_during_train=True,
    )
    trainer = LanguageTrainingSession(
        model, config, tokenizer, train_cfg, device=device,
    )

    def report(stats: dict):
        step = stats["step"]
        interval = max(1, min(50, n_steps))
        if step == 1 or step % interval == 0:
            val = f" val={stats['val_loss']:.3f}" if "val_loss" in stats else ""
            print(
                f"  step {step:4d} | loss={stats['loss']:.3f} "
                f"replay={stats['replay_loss']:.3f}{val} "
                f"||w||={stats['w_norm_mean']:.3f}"
            )

    trainer.fit_text(
        text,
        max_steps=n_steps,
        checkpoint_path='outputs/chat_model.pt',
        on_progress=report,
    )
    print(f"\n模型已保存: outputs/chat_model.pt")


def generate_response(model, tokenizer, prompt: str, n_new: int = 60,
                      temperature: float = 0.8, top_k: int = 5,
                      fast: bool = True) -> str:
    """生成回复"""
    # 把用户输入编码（未知字符用 0）
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    was_training = model.training
    with expert_integration_mode(model, use_rk4=not fast):
        generated = model.generate(
            prompt_ids, n_new=n_new, temperature=temperature, top_k=top_k,
        )
    if not was_training:
        model.eval()
    return tokenizer.decode(generated)


def chat(model, config, tokenizer, text, device: str):
    """交互对话循环"""
    print("\n" + "=" * 60)
    print("JspaceAI 对话模式")
    print("=" * 60)
    print("输入文本与模型对话，模型会持续学习你的输入。")
    print("命令:  /quit 退出  /save 保存  /reset 重置  /train N 预训练N步")
    print("=" * 60 + "\n")

    model.eval()
    learner = OnlineLanguageLearner(model, config, tokenizer, device=device)
    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break

        if not user_input:
            continue

        if user_input.startswith('/'):
            cmd = user_input.lower()
            if cmd in ('/quit', '/exit', '/q'):
                save_model(model, config, tokenizer)
                print(f"(状态已保存到 outputs/chat_model.pt)")
                print("再见")
                break
            elif cmd == '/save':
                save_model(model, config, tokenizer)
                print(f"(已保存到 outputs/chat_model.pt)")
                continue
            elif cmd == '/reset':
                model = JSpaceLanguageModel(config).to(device)
                learner = OnlineLanguageLearner(model, config, tokenizer, device=device)
                print("(模型已重置为随机初始化)")
                continue
            elif cmd.startswith('/train'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 50
                text = ensure_corpus(text)
                train(model, tokenizer, text, n, device)
                learner = OnlineLanguageLearner(model, config, tokenizer, device=device)
                model.eval()
                continue
            else:
                print("未知命令。可用: /quit /save /reset /train N")
                continue

        # 在线学习用户输入
        learn_stats = learner.learn_text(user_input)

        # 生成回复
        response = generate_response(
            model, tokenizer, user_input,
            n_new=40, temperature=0.8, top_k=5,
        )
        print(f"AI: {response}")
        if learn_stats is not None:
            print(f"   (学习 loss={learn_stats['loss']:.3f} replay={learn_stats['replay_loss']:.3f} step={learn_stats['step']})")


def generate_once(model, tokenizer, prompt: str, n_new: int = 80,
                  fast: bool = True):
    """一次性生成"""
    model.eval()
    response = generate_response(model, tokenizer, prompt, n_new=n_new,
                                  temperature=0.7, top_k=5, fast=fast)
    print(f"提示: {prompt}")
    print(f"生成: {response}")


def main():
    p = argparse.ArgumentParser(description='JspaceAI 对话版')
    p.add_argument('--mode', default='chat',
                   choices=['chat', 'train', 'generate'],
                   help='运行模式: chat=交互, train=预训练, generate=一次性生成')
    p.add_argument('--steps', type=int, default=600, help='train 模式步数')
    p.add_argument('--prompt', default='To be', help='generate 模式提示词')
    p.add_argument('--device', default='cpu', help='设备 (cpu/cuda/mps/auto)')
    p.add_argument('--n-new', type=int, default=80,
                   help='generate 模式生成的新字符数')
    p.add_argument('--accurate', action='store_true',
                   help='生成时使用 RK4（更慢但与训练配置一致）')
    args = p.parse_args()

    dev = args.device
    if dev == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu')

    model, config, tokenizer, text = load_or_init_model(dev)

    if args.mode == 'chat':
        chat(model, config, tokenizer, text, dev)
    elif args.mode == 'train':
        train(model, tokenizer, text, args.steps, dev)
    elif args.mode == 'generate':
        generate_once(model, tokenizer, args.prompt,
                      n_new=args.n_new, fast=not args.accurate)


if __name__ == '__main__':
    main()
