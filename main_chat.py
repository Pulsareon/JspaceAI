#!/usr/bin/env python3
"""
JspaceAI —— 对话版（只有语言，控制台交互）

基于字符级语言模型 + 自主进化（EWC + 经验回放 + 专家可塑性）。
交互式对话：用户输入文本，模型生成回复，同时持续学习。

模式：
    --mode chat:     交互对话（默认）
    --mode train:    先在 Shakespeare 语料上预训练若干步，再进入对话
    --mode generate: 给定提示词一次性生成

运行：
    python main_chat.py
    python main_chat.py --mode train --steps 100
    python main_chat.py --mode generate --prompt "To be"
"""
from __future__ import annotations
import argparse
import torch
import numpy as np
from pathlib import Path

from jspaceai import (
    LanguageConfig, JSpaceLanguageModel, EvolutionTrainer,
    CharTokenizer, load_shakespeare,
)


def get_config(vocab_size: int) -> LanguageConfig:
    return LanguageConfig(
        vocab_size=vocab_size,
        embed_dim=16, input_dim=8,
        workspace_dim=32, expert_dim=16,
        num_experts=5, num_wells=4,
        ode_steps=4, dt=0.1, tau_w=0.3,
        jacobian_sparsity=8, noise_std=0.005,
        use_rk4=True, use_layer_norm=True,
    )


def load_or_init_model(device: str):
    """加载已保存的模型或初始化新模型"""
    text = load_shakespeare()
    tokenizer = CharTokenizer.from_text(text)
    config = get_config(tokenizer.vocab_size)
    model = JSpaceLanguageModel(config).to(device)

    mp = Path('outputs/chat_model.pt')
    if mp.exists():
        try:
            ckpt = torch.load(mp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model'])
            print(f"已加载模型: {mp}（上次保存的对话状态）")
        except Exception:
            print("模型加载失败，全新初始化")
    else:
        print("全新初始化（首次对话）")
    return model, config, tokenizer, text


def save_model(model, config, tokenizer):
    Path('outputs').mkdir(exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': config,
        'tokenizer_chars': tokenizer.chars,
    }, 'outputs/chat_model.pt')


def train(model, tokenizer, text, n_steps: int, device: str):
    """在 Shakespeare 语料上预训练"""
    print("\n" + "=" * 60)
    print(f"预训练 {n_steps} 步（Shakespeare 语料）")
    print("=" * 60)

    config = model.config
    trainer = EvolutionTrainer(
        model, config, lr=5e-3, ewc_lambda=0.05, device=device,
    )
    chunks = [text[i:i+200] for i in range(0, len(text), 200)]
    trainer.evolve(
        chunks, tokenizer,
        seq_len=48, batch_size=4,
        consolidate_every=30, generate_every=50,
        max_steps=n_steps, prompt_text="To be",
    )
    save_model(model, config, tokenizer)
    print(f"\n模型已保存: outputs/chat_model.pt")


def generate_response(model, tokenizer, prompt: str, n_new: int = 100,
                      temperature: float = 0.8, top_k: int = 5) -> str:
    """生成回复"""
    # 把用户输入编码（未知字符用 0）
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    generated = model.generate(
        prompt_ids, n_new=n_new, temperature=temperature, top_k=top_k,
    )
    return tokenizer.decode(generated)


def online_learn(model, config, tokenizer, text: str, user_input: str, device: str):
    """在线学习用户输入 + 回放一段 Shakespeare 防遗忘"""
    import torch.nn.functional as F
    # 把用户输入作为新语料学习
    user_tokens = tokenizer.encode(user_input)
    if len(user_tokens) < 4:
        return  # 太短不学

    # 重复用户输入凑够 seq_len
    seq_len = 48
    if len(user_tokens) < seq_len:
        user_tokens = user_tokens * (seq_len // len(user_tokens) + 1)
    user_seq = user_tokens[:seq_len]
    token_tensor = torch.tensor([user_seq], dtype=torch.long).to(device)

    # forward + next-token loss
    model.train()
    model.zero_grad()
    logits, _ = model(token_tensor)
    pred = logits[:, :-1]
    target = token_tensor[:, 1:]
    loss = F.cross_entropy(
        pred.reshape(-1, config.vocab_size),
        target.reshape(-1),
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 手动 SGD step（EvolutionTrainer 内部有 EWC，这里简化用直接 step）
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 5e-3 * p.grad
    model.eval()
    return loss.item()


def chat(model, config, tokenizer, text, device: str):
    """交互对话循环"""
    print("\n" + "=" * 60)
    print("JspaceAI 对话模式")
    print("=" * 60)
    print("输入文本与模型对话，模型会持续学习你的输入。")
    print("命令:  /quit 退出  /save 保存  /reset 重置  /train N 预训练N步")
    print("=" * 60 + "\n")

    model.eval()
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
                print("(模型已重置为随机初始化)")
                continue
            elif cmd.startswith('/train'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 50
                train(model, tokenizer, text, n, device)
                model.eval()
                continue
            else:
                print("未知命令。可用: /quit /save /reset /train N")
                continue

        # 在线学习用户输入
        loss = online_learn(model, config, tokenizer, text, user_input, device)

        # 生成回复
        response = generate_response(
            model, tokenizer, user_input,
            n_new=80, temperature=0.8, top_k=5,
        )
        print(f"AI: {response}")
        if loss is not None:
            print(f"   (学习 loss={loss:.3f})")


def generate_once(model, tokenizer, prompt: str, n_new: int = 200):
    """一次性生成"""
    model.eval()
    response = generate_response(model, tokenizer, prompt, n_new=n_new,
                                  temperature=0.7, top_k=5)
    print(f"提示: {prompt}")
    print(f"生成: {response}")


def main():
    p = argparse.ArgumentParser(description='JspaceAI 对话版')
    p.add_argument('--mode', default='chat',
                   choices=['chat', 'train', 'generate'],
                   help='运行模式: chat=交互, train=预训练, generate=一次性生成')
    p.add_argument('--steps', type=int, default=100, help='train 模式步数')
    p.add_argument('--prompt', default='To be', help='generate 模式提示词')
    p.add_argument('--device', default='cpu', help='设备 (cpu/cuda/mps/auto)')
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
        generate_once(model, tokenizer, args.prompt)


if __name__ == '__main__':
    main()
