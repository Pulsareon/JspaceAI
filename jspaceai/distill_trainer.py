"""蒸馏训练器——把小模型理解能力迁移到 JspaceAI"""
from __future__ import annotations
import torch, torch.nn.functional as F, numpy as np


class DistillationTrainer:
    """蒸馏训练器

    把小模型的理解能力迁移到 JspaceAI 编码器。
    训练后编码器离线也有理解能力，小模型可以断开。
    """

    def __init__(self, model_v2, encoder, texts, device='cpu'):
        self.model = model_v2.to(device)
        self.encoder = encoder.to(device)
        self.texts = texts
        self.device = device
        params = list(model_v2.parameters()) + list(encoder.proj.parameters())
        self.optimizer = torch.optim.Adam(params, lr=1e-3)
        self.history = []

    def train_step(self, text_batch):
        # 1. 小模型表征（target）
        target = None
        if self.encoder._real_model is not None:
            with torch.no_grad():
                inputs = self.encoder._tokenizer(
                    text_batch, return_tensors="pt",
                    truncation=True, max_length=64, padding=True
                ).to(self.device)
                outputs = self.encoder._model(**inputs)
                target = outputs.last_hidden_state.mean(dim=1)

        # 2. 编码器输出
        encoded = self.encoder(text_batch)

        # 3. workspace 处理
        xs = encoded.unsqueeze(1)  # (batch, 1, input_dim)
        state = self.model.init_state(xs.shape[0], self.device)
        w_out, info = self.model(xs, state)

        # 4. 蒸馏 loss
        if target is not None:
            target_proj = self.encoder.proj(target.to(self.device)).detach()
            distill_loss = F.mse_loss(encoded, target_proj)
        else:
            # 自监督
            if encoded.shape[0] > 1:
                distill_loss = F.mse_loss(encoded[:-1], encoded[1:].detach())
            else:
                distill_loss = torch.tensor(0.0, device=self.device)

        # 5. workspace 稳定性（目标 ||w||≈1）
        w_norm = info['w_norm']
        stability_loss = F.mse_loss(w_norm, torch.ones_like(w_norm))

        # 6. 总 loss
        total_loss = distill_loss + 0.1 * stability_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.history.append({
            'loss': total_loss.item(),
            'distill': distill_loss.item(),
            'stability': stability_loss.item(),
            'w_norm': w_norm.mean().item(),
        })
        return self.history[-1]

    def train(self, n_steps=100, batch_size=4, verbose=True):
        for step in range(n_steps):
            batch = np.random.choice(self.texts, size=min(batch_size, len(self.texts)),
                                    replace=True).tolist()
            info = self.train_step(batch)
            if verbose and (step+1) % 20 == 0:
                print(f"  step {step+1:4d} | loss {info['loss']:.4f} | "
                      f"distill {info['distill']:.4f} | "
                      f"||w|| {info['w_norm']:.3f}")
        return self.history
