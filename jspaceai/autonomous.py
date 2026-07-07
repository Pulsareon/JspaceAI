"""
自主心智（AutonomousMind）—— 智慧最重要的能力

四个核心能力：
    1. CuriosityDrive: 好奇心驱动的主动探索（内在奖励）
    2. PersistentState: 跨会话状态持久化（海洋不蒸发）
    3. SelfModel: 自我模型（知道自己会什么不会什么）
    4. MetaLearner: 元学习（学会如何学习）

合起来 = AutonomousMind，永不停止的自主进化。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from collections import deque


class CuriosityDrive(nn.Module):
    """好奇心驱动——基于预测误差的内在奖励

    模型有世界模型预测下一状态，预测误差=好奇心=内在奖励。
    模型被驱动去探索"预测不准"的区域。学会后好奇心降低，转向新区域。
    """

    def __init__(self, workspace_dim: int, action_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.world_model = nn.Sequential(
            nn.Linear(workspace_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )
        self.state_history: deque = deque(maxlen=500)
        self.prediction_error_ema = 0.1

    def predict_next(self, w, action):
        return self.world_model(torch.cat([w, action], dim=-1))

    def compute_curiosity(self, w_current, action, w_next):
        with torch.no_grad():
            w_pred = self.predict_next(w_current, action)
            pred_error = F.mse_loss(w_pred, w_next).item()
            w_np = w_current[0].cpu().numpy()
            novelty = self._compute_novelty(w_np)
            progress = max(0, pred_error - self.prediction_error_ema * 0.9)
            self.prediction_error_ema = 0.95 * self.prediction_error_ema + 0.05 * pred_error
            curiosity = progress + 0.3 * novelty
            self.state_history.append(w_np.copy())
            return curiosity

    def _compute_novelty(self, w):
        if len(self.state_history) < 5:
            return 1.0
        history = list(self.state_history)[-100:]
        distances = [np.linalg.norm(w - h) for h in history]
        return float(min(1.0, min(distances) / 2.0))

    def train_world_model(self, w_current, action, w_next):
        w_pred = self.predict_next(w_current, action.detach())
        return F.mse_loss(w_pred, w_next.detach())


class PersistentState:
    """跨会话状态持久化——让海洋不蒸发

    保存 workspace + 专家状态 + 海马体 + 基底神经节 + 好奇心历史 + 自我模型。
    """

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.save_dir / 'mind_state.json'
        self.tensors_file = self.save_dir / 'mind_tensors.npz'

    def save(self, state: dict):
        tensors = {}
        if 'w' in state:
            tensors['w'] = state['w'].cpu().numpy()
        if 'm' in state:
            for i, m in enumerate(state['m']):
                if m is not None:
                    tensors[f'm_{i}'] = m.cpu().numpy()
        if 'basal_ganglia' in state and state['basal_ganglia'] is not None:
            tensors['bg_weights'] = state['basal_ganglia']
        if 'curiosity_history' in state:
            tensors['curiosity_history'] = np.array(state['curiosity_history'])
        if tensors:
            np.savez(self.tensors_file, **tensors)

        json_state = {
            'step_count': state.get('step_count', 0),
            'total_runtime': state.get('total_runtime', 0.0),
            'self_model': state.get('self_model', {}),
            'saved_at': time.time(),
        }
        self.state_file.write_text(json.dumps(json_state, indent=2, default=str))

    def load(self) -> Optional[dict]:
        if not self.state_file.exists():
            return None
        result = {}
        if self.tensors_file.exists():
            data = np.load(self.tensors_file, allow_pickle=True)
            if 'w' in data:
                result['w'] = torch.tensor(data['w'])
            ms = {}
            for key in data.files:
                if key.startswith('m_'):
                    idx = int(key.split('_')[1])
                    ms[idx] = torch.tensor(data[key])
            if ms:
                result['m'] = [ms[i] for i in sorted(ms.keys())]
            if 'bg_weights' in data:
                result['basal_ganglia'] = data['bg_weights']
            if 'curiosity_history' in data:
                result['curiosity_history'] = data['curiosity_history'].tolist()
        json_state = json.loads(self.state_file.read_text())
        result.update(json_state)
        return result


class SelfModel:
    """自我模型——知道自己会什么、不会什么

    对每个能力域维护置信度（0-1），通过历史成功率更新。
    不知道时主动学习（好奇心驱动）。
    """

    def __init__(self, capabilities=None):
        if capabilities is None:
            capabilities = ['visual', 'audio', 'text', 'motor_mouse',
                           'motor_keyboard', 'memory', 'prediction']
        self.capabilities = capabilities
        self.confidence = {c: 0.0 for c in capabilities}
        self.attempts = {c: 0 for c in capabilities}
        self.recent_results = {c: deque(maxlen=20) for c in capabilities}

    def record_attempt(self, capability, success):
        if capability not in self.confidence:
            return
        self.attempts[capability] += 1
        self.recent_results[capability].append(success)
        recent = list(self.recent_results[capability])
        if recent:
            weights = np.linspace(0.5, 1.0, len(recent))
            self.confidence[capability] = float(np.average(recent, weights=weights))

    def get_weakness(self):
        return min(self.confidence, key=self.confidence.get)

    def get_strength(self):
        return max(self.confidence, key=self.confidence.get)

    def knows(self, capability, threshold=0.5):
        return self.confidence.get(capability, 0.0) > threshold

    def summary(self):
        return {
            'capabilities': dict(self.confidence),
            'attempts': dict(self.attempts),
            'strength': self.get_strength(),
            'weakness': self.get_weakness(),
        }


class MetaLearner:
    """元学习——学会如何学习

    每个能力域有自适应学习率：进步快→加快，停滞→减慢。
    记录什么学习策略有效。
    """

    def __init__(self, capabilities=None):
        if capabilities is None:
            capabilities = ['visual', 'audio', 'text', 'motor_mouse',
                           'motor_keyboard', 'memory', 'prediction']
        self.learning_rates = {c: 1e-3 for c in capabilities}
        self.loss_history = {c: deque(maxlen=20) for c in capabilities}
        self.strategy_scores = {
            'predict_next': 0.5, 'replay': 0.5,
            'explore': 0.5, 'imitate': 0.5,
        }

    def get_lr(self, capability):
        return self.learning_rates.get(capability, 1e-3)

    def record_loss(self, capability, loss):
        if capability not in self.loss_history:
            return
        self.loss_history[capability].append(loss)
        history = list(self.loss_history[capability])
        if len(history) < 5:
            return
        recent_avg = np.mean(history[-5:])
        old_avg = np.mean(history[-10:-5]) if len(history) >= 10 else recent_avg
        improvement = (old_avg - recent_avg) / max(old_avg, 1e-8)
        lr = self.learning_rates[capability]
        if improvement > 0.05:
            lr *= 1.1
        elif improvement < 0.01:
            lr *= 0.9
        self.learning_rates[capability] = max(1e-5, min(1e-2, lr))

    def best_strategy(self):
        return max(self.strategy_scores, key=self.strategy_scores.get)

    def reward_strategy(self, strategy, reward):
        if strategy in self.strategy_scores:
            self.strategy_scores[strategy] = (
                0.9 * self.strategy_scores[strategy] + 0.1 * reward
            )


class AutonomousMind:
    """自主心智——永不停止的自主进化

    整合好奇心 + 持久化 + 自我模型 + 元学习。
    核心循环：感知→自我评估→好奇探索→行动→观察→学习→存盘
    """

    def __init__(self, agent, save_dir='outputs/mind', device='cpu'):
        self.agent = agent
        self.device = device
        self.config = agent.config

        self.curiosity = CuriosityDrive(
            workspace_dim=self.config.workspace_dim, action_dim=5,
        ).to(device)
        self.persistence = PersistentState(Path(save_dir))
        self.self_model = SelfModel()
        self.meta_learner = MetaLearner()

        self.step_count = 0
        self.total_runtime = 0.0
        self.start_time = time.time()
        self.running = False
        self.curiosity_history = deque(maxlen=1000)
        self.curiosity_optimizer = torch.optim.Adam(
            self.curiosity.parameters(), lr=1e-3
        )

        self._load_state()

    def _load_state(self):
        state = self.persistence.load()
        if state is None:
            print("  [心智] 全新启动")
            return
        print(f"  [心智] 恢复状态: step={state.get('step_count', 0)}")
        if 'w' in state:
            self.agent.state['w'] = state['w'].to(self.device)
        if 'm' in state:
            for i, m in enumerate(state['m']):
                if m is not None and i < len(self.agent.state['m']):
                    self.agent.state['m'][i] = m.to(self.device)
        if 'basal_ganglia' in state and hasattr(self.agent, 'basal_ganglia'):
            self.agent.basal_ganglia.action_weights = state['basal_ganglia']
        self.step_count = state.get('step_count', 0)
        self.total_runtime = state.get('total_runtime', 0.0)

    def save_state(self):
        state = {
            'w': self.agent.state['w'],
            'm': self.agent.state['m'],
            'basal_ganglia': getattr(self.agent.basal_ganglia, 'action_weights', None)
                             if hasattr(self.agent, 'basal_ganglia') else None,
            'curiosity_history': list(self.curiosity.state_history),
            'step_count': self.step_count,
            'total_runtime': self.total_runtime + (time.time() - self.start_time),
            'self_model': {'confidence': self.self_model.confidence},
        }
        self.persistence.save(state)

    def step(self) -> dict:
        # 1. 感知 + 思考
        sensory_data = self.agent.perceive()
        w_before = self.agent.state['w'].clone()
        weakness = self.self_model.get_weakness()
        strength = self.self_model.get_strength()

        # 2. 行动
        action_info = self.agent.decide_and_act(w_before, sensory_data.get('modality', 'idle'))
        w_after, modality = self.agent.think(sensory_data)

        # 3. 好奇心
        action_tensor = torch.tensor(action_info['action_params'],
                                    dtype=torch.float32).unsqueeze(0).to(self.device)
        curiosity_reward = self.curiosity.compute_curiosity(w_before, action_tensor, w_after)
        self.curiosity_history.append(curiosity_reward)

        # 4. 训练世界模型
        world_loss = self.curiosity.train_world_model(w_before, action_tensor, w_after)
        self.curiosity_optimizer.zero_grad()
        world_loss.backward()
        self.curiosity_optimizer.step()

        # 5. 评估成功度
        w_stability = 1.0 - min(1.0, abs(w_after.norm().item() - w_before.norm().item()))
        success = (0.3 * float(action_info['executed']) +
                   0.4 * min(1.0, curiosity_reward) + 0.3 * w_stability)

        # 6. 更新自我模型
        cap_map = {'image': 'visual', 'screen': 'visual', 'audio': 'audio',
                   'text': 'text', 'keyboard': 'text', 'mouse': 'motor_mouse', 'idle': 'prediction'}
        cap = cap_map.get(modality, 'prediction')
        self.self_model.record_attempt(cap, success)
        self.self_model.record_attempt('prediction', 1.0 - min(1.0, world_loss.item()))

        # 7. 元学习
        self.meta_learner.record_loss(cap, world_loss.item())
        if curiosity_reward > 0.3:
            self.meta_learner.reward_strategy('explore', curiosity_reward)

        # 8. 记忆 + 基底神经节
        self.agent.remember(w_after, {
            'modality': modality, 'curiosity': curiosity_reward,
            'success': success, 'step': self.step_count,
        })
        self.agent.learn(w_after, np.array(action_info['action_params']), reward=curiosity_reward)

        self.step_count += 1
        return {
            'step': self.step_count, 'modality': modality,
            'w_norm': w_after.norm().item(), 'curiosity': curiosity_reward,
            'world_loss': world_loss.item(), 'success': success,
            'weakness': weakness, 'strength': strength,
            'self_confidence': dict(self.self_model.confidence),
            'best_strategy': self.meta_learner.best_strategy(),
            'memory_count': self.agent.hippocampus.size() if self.agent.hippocampus else 0,
        }

    def run(self, n_steps=100, interval=0.2, save_every=50, on_step=None):
        self.running = True
        self.start_time = time.time()
        self.agent.senses.start()
        print(f"\n自主心智启动 | 总步数: {self.step_count} | 保存间隔: {save_every}")
        print("=" * 60)

        log = []
        try:
            for _ in range(n_steps):
                if not self.running:
                    break
                info = self.step()
                log.append(info)
                if on_step:
                    on_step(info)
                elif info['step'] % 10 == 0:
                    print(f"  step {info['step']:4d} | mod {info['modality']:8s} | "
                          f"||w|| {info['w_norm']:.3f} | curio {info['curiosity']:.3f} | "
                          f"success {info['success']:.2f} | weak={info['weakness']} | "
                          f"mem {info['memory_count']}")
                if info['step'] % save_every == 0:
                    self.save_state()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.running = False
            self.agent.senses.stop()
            if hasattr(self.agent, 'audio_actuator'):
                self.agent.audio_actuator.stop()
            self.save_state()
        self.total_runtime += time.time() - self.start_time
        return log

    def introspect(self) -> str:
        """内省——自我报告"""
        sm = self.self_model.summary()
        avg_curio = np.mean(list(self.curiosity_history)) if self.curiosity_history else 0
        report = f"=== 自主心智内省 ===\n步数: {self.step_count}\n运行: {self.total_runtime:.0f}s\n"
        report += f"记忆: {self.agent.hippocampus.size() if self.agent.hippocampus else 0}\n"
        report += f"平均好奇心: {avg_curio:.3f}\n\n自我认知:\n"
        for cap, conf in sm['capabilities'].items():
            bar = '█' * int(conf * 20)
            report += f"  {cap:15s}: {conf:.2f} {bar}\n"
        report += f"\n最强: {sm['strength']}\n最弱: {sm['weakness']}\n"
        report += f"最佳策略: {self.meta_learner.best_strategy()}\n"
        return report
