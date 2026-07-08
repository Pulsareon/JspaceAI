# JspaceAI

**全局工作空间 + J-space 广播的智慧系统**

两个入口：
1. **全部接入** (`main.py`)：具身 Agent + 完整神经系统 + 5 感官 + 4 输出 + 自主心智
2. **只有对话** (`main_chat.py`)：字符级语言模型 + 交互对话 + 在线学习

基于第一性原理推导的智慧架构：不是 Transformer，不是 RNN，而是 **ODE 动力系统 + 并行专家 + J-space 工作空间广播 + 在线进化**。

## 核心思想

智慧 = 系统内部维持对世界的模型 M，可操作、可修正、可指导行动。

从第一性原理推导出 M 必须满足的性质，每一性质对应一个实现选择：

| 性质 | 实现 |
|---|---|
| 高维 | 向量空间 $\mathbb{R}^n$ |
| 内部动力学 | ODE $\dot{m} = -\nabla U(m) + \text{broadcast}$ |
| 输入敏感但不决定 | 加性扰动 |
| 可读出 | 流形投影 + 阈值门控 |
| 可修正 | 参数化 + backprop |
| 工作空间结构 | 分块 + J-space 广播 |

## 架构

```
┌─────────────────────────────────────┐
│  专家池 E_1..E_5（并行）              │
│  每个 m_i 有多井势能 → 内部"思考"      │
│  J_i 稀疏 Jacobian → 选择性接收广播    │
└─────────────────────────────────────┘
              ↓ 写入（注意力 α_i 选择）
┌─────────────────────────────────────┐
│  工作空间 w（J-space）                │
│  τ_w · dw/dt = -w + Σ α_i · P_i(m_i) │
│  有自己的动力学 → 持续演化            │
└─────────────────────────────────────┘
              ↓ 读出
        预测头 Q(w) → 预测下一时刻世界
```

**与 Transformer 的对比**：

| | Transformer | JspaceAI |
|---|---|---|
| 内部状态 | 离散、无状态 | 连续 ODE 动力学 |
| 工作空间 | 隐式（residual stream） | 显式分块 + 广播 |
| 专家协作 | 无（或 MoE 但孤立） | J-space 路由 |
| 学习信号 | 下一 token | 下一时刻世界状态 |
| 自发输出 | 否（被输入触发） | 是（阈值门控） |

## 安装

```bash
pip install -r requirements.txt
```

## 运行

项目只有两个入口：

| 入口 | 用途 |
|---|---|
| `main.py` | 全部接入：具身 Agent + 完整神经系统 + 5 感官 + 4 输出 + 自主心智 |
| `main_chat.py` | 只有对话：字符级语言模型 + 交互对话 + 在线学习 |

### 全部接入版（`main.py`）

```bash
# 子系统自检（权限 + I/O + 模型 + 海马体回忆）
python main.py --mode test

# 实时具身循环 + 自主心智（默认安全：不控制鼠标键盘）
python main.py --mode live --steps 100

# 安全模式（更高动作门控阈值，不显示屏幕输出）
python main.py --mode safe --steps 50

# 允许鼠标键盘输出（需谨慎，加 --unsafe）
python main.py --mode live --steps 200 --unsafe
```

接入全部 5 个输入通道 + 4 个输出 + 神经系统：
- **输入**：摄像头(OpenCV) + 麦克风(PortAudio) + 屏幕(mss) + 键盘(pynput) + 鼠标(pynput)
- **输出**：鼠标 + 键盘 + 音频(PortAudio) + 屏幕(OpenCV 窗口)
- **神经系统**：小脑(前向+逆模型) / 中枢神经(反射弧+门控) / 海马体(向量检索) / 基底神经节(Q-learning)
- **自主心智**：好奇心驱动 + 状态持久化 + 自我模型 + 元学习

完整的人类神经系统对应：

| 人类系统 | 功能 | 实现 |
|---|---|---|
| 大脑皮层 | 认知、语言 | workspace + 12 专家 |
| 小脑 | 运动协调 | Cerebellum（前向模型+逆模型） |
| 中枢神经 | 动作门控 | CentralNervousSystem（反射弧+抑制） |
| 海马体 | 情景记忆 | Hippocampus（向量检索） |
| 基底神经节 | 动作选择 | BasalGanglia（Q-learning） |
| 眼/耳 | 视听觉 | 摄像头+麦克风 |
| 口 | 说话 | AudioActuator（扬声器） |
| 手 | 操作 | MouseActuator + KeyboardActuator |
| 皮肤 | 触觉 | （预留接口） |

感知-思考-行动闭环：
1. 感知（五感）→ 2. 思考（workspace）→ 3. 回忆（海马体）
→ 4. 决策（基底神经节）→ 5. 精化（小脑）→ 6. 门控（中枢神经）
→ 7. 行动（手足口）→ 8. 学习（更新所有系统）

状态保存在 `outputs/mind/`：workspace + 海马体 + 基底神经节 + 自我模型，再次运行从上次继续。

**macOS 权限**：键盘/鼠标监听需要在 系统设置 → 隐私与安全 → 辅助功能 中授权终端。

### 对话版（`main_chat.py`）

```bash
# 交互对话（默认）
python main_chat.py

# 先预训练再对话
python main_chat.py --mode train --steps 100
python main_chat.py --mode chat

# 一次性生成
python main_chat.py --mode generate --prompt "To be"

# 控制生成长度；默认使用快速 Euler 推理，--accurate 改用 RK4
python main_chat.py --mode generate --prompt "学而时习之" --n-new 40
python main_chat.py --mode generate --prompt "学而时习之" --accurate
```

对话特性：
- **在线学习**：每次用户输入都会被模型学习（next-token loss + 梯度更新）
- **持续进化**：EWC + 经验回放 + 专家可塑性防灾难性遗忘
- **状态持久化**：对话状态保存到 `outputs/chat_model.pt`，再次启动从上次继续
- **命令**：`/quit` 退出 `/save` 保存 `/reset` 重置 `/train N` 预训练N步

对话模式示例：
```
你: Hello
AI: ...
你: /save
你: /train 50
你: /quit
```

## 测试

```bash
python -m unittest discover -s tests -v
```

## 外挂模块系统（可热插拔）

```python
from jspaceai import ModuleDock, SmallModelModule, KnowledgeBaseModule, ToolModule

dock = ModuleDock(workspace_dim=64)
dock.register('llm', SmallModelModule(model_name='gpt2'))
dock.register('kb', KnowledgeBaseModule())
dock.register('tools', ToolModule())

# 热插上
dock.connect('llm')
dock.connect('tools')

# 使用
result = dock.query('llm', 'hello world')
calc = dock.query('tools', {'tool': 'calc', 'args': ['1+1']})  # → 2

# 随时拔掉（心智不停，只是该能力不可用）
dock.disconnect('llm')
dock.query('llm', 'hello')  # → None

# 状态查询
dock.status()  # 各模块连接状态
```

三种内置外挂：
- **SmallModelModule**：外挂小模型（GPT-2/Qwen 等），提供语言能力
- **KnowledgeBaseModule**：向量检索知识库
- **ToolModule**：可调用工具（计算器、搜索等）

核心原则：**心智不依赖外挂**。断开任何外挂，心智继续工作，只是"知道得少"。

## 核心实现特性

`JSpaceConfig` 支持以下可配置项（默认值已开启推荐配置）：

1. **RK4 积分**（`use_rk4=True`）：比 Euler 稳定，workspace 范数从 ~0.3 提升到 ~5.6
2. **LayerNorm**（`use_layer_norm=True`）：workspace 和专家状态都加 LayerNorm，防止长期衰减
3. **势能系数归一化**（`well_coeff=None` 自动 `1/sqrt(num_wells)`）：避免井数多时 softplus 项主导梯度导致专家状态发散
4. **异构专家支持**：`Expert` 构造可传入 `encoder` 参数，让不同模态专家有独立编码器
5. **轨迹记录**：`forward(xs, record_trajectory=True)` 返回 `info['w_trajectory']`，供 J-lens 训练

```python
from jspaceai import JSpaceConfig, JSpaceModel

config = JSpaceConfig(
    workspace_dim=32,
    num_experts=5,
    use_rk4=True,          # RK4 积分
    use_layer_norm=True,   # LayerNorm 防衰减
)
model = JSpaceModel(config)
preds, info = model(xs, record_trajectory=True)  # info['w_trajectory'] 可用
```

## 跨平台支持

支持 macOS / Windows / Linux：

| 功能 | macOS | Windows | Linux |
|---|---|---|---|
| 摄像头 | OpenCV | OpenCV | OpenCV |
| 麦克风 | PortAudio | PortAudio | PortAudio |
| 屏幕 | mss | mss | mss/X11 |
| 键盘监听 | pynput（需辅助功能权限） | pynput | pynput（需 X11） |
| 鼠标监听 | pynput（需辅助功能权限） | pynput | pynput（需 X11） |
| 鼠标控制 | pynput | pynput | pynput |
| 键盘控制 | pynput | pynput | pynput |
| 音频播放 | PortAudio | PortAudio | PortAudio |

## 自主进化机制

模型不只是"训练一次"，而是**天生支持持续进化**：

1. **在线学习**：每个 forward 都累积梯度并更新参数，模型永不"训练完成"
2. **EWC（Elastic Weight Consolidation）**：保护重要参数防灾难性遗忘
   - 周期性计算 Fisher 信息，锚定重要参数
   - 新知识只能修改不重要的参数
3. **经验回放（Experience Replay）**：定期回放旧序列，防遗忘
4. **专家可塑性（Expert Plasticity）**：追踪专家专业化，鼓励新知识流向空闲专家

进化循环可以**无限运行**——持续喂入新文本，模型持续进化。

## 输出

### 全部接入版
- `outputs/embodied_live.png` — ||w||/模态分布/好奇心/自我模型/世界模型loss/海马体
- `outputs/mind/` — 自主心智状态（跨会话持久化）

### 对话版
- `outputs/chat_model.pt` — 对话模型状态（跨会话持久化）

## 基于 Anthropic 2026 论文的设计

基于 [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) 的发现：

1. **J-lens 探针**：学习线性映射 L: workspace → vocab，观测每个子步 workspace "准备说什么"（`jlens.py` 提供）
2. **三层分层**：sensory（子步0）→ workspace（子步1-2）→ motor（子步3）
3. **Directed Modulation**：注入概念向量 v_concept 到 workspace，验证 workspace 是因果中介
4. **Selectivity 验证**：ablate workspace top-k J-lens 方向，对比自动任务 vs 记忆任务

## 实验结果

### 语言版
- Loss 从 3.95 → 2.40，下降 39%
- 生成从纯乱码进化为类语言结构（含换行、常见词片段）
- 专家按字符类型分工（元音/辅音/结构字符）

### 具身版
- 实时 5 通道感知（摄像头+麦克风+屏幕+键盘+鼠标）全部采集成功
- 感知-思考-行动闭环完整运行
- 海马体回忆正常（sim 0.99+）
- 小脑计算动作参数，基底神经节选择动作，中枢神经门控

## 文件结构

```
JspaceAI/
├── main.py                  # 全部接入（具身 + 自主心智）
├── main_chat.py             # 只有对话
├── requirements.txt
├── README.md
└── jspaceai/
    ├── __init__.py
    ├── core.py              # 核心：Expert + JSpaceWorkspace + JSpaceModel（RK4+LayerNorm）
    ├── baselines.py         # 对比基线：FlatBaseline
    ├── task.py              # 玩具任务：多模态连续序列
    ├── trainer.py           # 训练器：预测学习
    ├── language_data.py     # 字符级 tokenizer + Shakespeare 语料
    ├── language_model.py    # 语言版 + EWC + 经验回放 + 专家可塑性
    ├── jlens.py             # J-lens 探针 + WorkspaceAblator + DirectedModulation
    ├── multimodal.py        # 多模态核心（6 模态编解码 + 12 专家）
    ├── realtime.py          # 摄像头 + 麦克风 + SensoryMotorLoop
    ├── desktop.py           # 屏幕 + 键盘鼠标 + FullSensoryStream
    ├── platform.py          # 跨平台抽象 + 权限检查
    ├── embodied.py          # 神经系统（小脑/海马体/基底神经节/中枢神经）+ EmbodiedAgent
    ├── autonomous.py        # 自主心智（好奇心/持久化/自我模型/元学习）
    ├── modules.py           # 可热插拔外挂模块
    └── evolution.py         # 自主进化训练器
```

## 数学细节

### 专家动力学

$$\dot{m}_i = -\nabla U_i(m_i) + J_i \cdot w + P_i^{in} \cdot x + \xi$$

势能 $U_i(m_i) = \frac{1}{2}\|m_i\|^2 - \frac{c}{2}\sum_k \text{softplus}(a_k \cdot m_i + b_k)$，其中 $c = 1/\sqrt{\text{num\_wells}}$ 防梯度被 softplus 主导。

多井结构让专家在吸引子之间漫游 = 内部"思考"。

### 工作空间动力学

$$\tau_w \cdot \dot{w} = -w + \sum_i \alpha_i \cdot P_i^{out}(m_i)$$

$$\alpha_i = \text{softmax}(\langle q, P_i^{out}(m_i) \rangle), \quad q = \text{MLP}(x, w)$$

### J-space 路由

$J_i$ 是稀疏矩阵（top-k 选择），每个专家只对工作空间的少数方向敏感 → 选择性广播。

### 学习目标

$$\mathcal{L} = \mathbb{E}\left[\|x_{t+\Delta t} - Q(w_t)\|^2\right]$$

自监督——只需时间序列，不需标签。

## 局限性

这是**最小验证版本**，不是生产系统：

- 玩具任务（8 维序列），不是语言/图像
- 专家数 5，不是大规模
- 对话版字符级，不是 subword

**下一步扩展方向**：
1. 用 adjoint method 替换 RK4（Neural ODE 路线，更低显存）
2. Scale 到更大任务（MNIST 连续预测 → 语言建模）
3. 加输出门控的真正"自发输出"（非每步都预测）
4. 接 Whisper/VITS 真实语音输入输出
