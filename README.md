# JspaceAI

**全局工作空间 + J-space 广播的智慧系统**

两个版本：
1. **连续序列版** (`main.py`)：验证架构在多模态序列预测上的优势
2. **语言版 + 自主进化** (`main_language.py`)：字符级语言建模 + 边推理边学习

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

### 连续序列版（验证架构）

```bash
python main.py
python main.py --steps 2000 --device cuda
```

### v2 版（J-lens + Selectivity + Directed Modulation）

```bash
python main_v2.py
python main_v2.py --steps 200 --device mps
```

基于 Anthropic 2026 J-space 论文优化，新增：
- **J-lens**：观测模型内部每个 ODE 子步的"想法"
- **Directed Modulation**：注入概念向量到 workspace，定向改变输出
- **Selectivity 验证**：ablate workspace，对比自动任务 vs 记忆任务

### 多模态版（摄像头 + 麦克风 + 扬声器）

```bash
python main_multimodal.py --mode train --steps 300
python main_multimodal.py --mode live --steps 50
python main_multimodal.py --mode eval
```

原生支持图像/音频/视频/文本四种模态，8 个专家分工。

### 全感官版（摄像头 + 麦克风 + 屏幕 + 键盘 + 鼠标）

```bash
# 测试所有 I/O 通道
python main_full_sensory.py --mode test

# 实时全感官感知循环
python main_full_sensory.py --mode live --steps 50
```

接入全部 5 个输入通道，12 个专家分工：
- **摄像头**（OpenCV）：环境视觉
- **麦克风**（PortAudio）：听觉输入
- **屏幕**（mss）：屏幕捕获，看到自己被显示在哪里
- **键盘**（pynput）：用户输入感知
- **鼠标**（pynput）：用户注意力追踪
- **扬声器**（PortAudio）：音频输出

**macOS 权限**：键盘/鼠标监听需要在 系统设置 → 隐私与安全 → 辅助功能 中授权终端。

### 具身 Agent 版（完整神经系统）

```bash
# 测试各子系统
python main_embodied.py --mode test

# 实时具身循环（鼠标键盘音频屏幕全输出）
python main_embodied.py --mode live --steps 50

# 安全模式（只感知不执行鼠标键盘动作）
python main_embodied.py --mode safe --steps 30
```

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

### 守护进程（后台持续运行）

```bash
# 启动后台守护进程（静默运行，不干扰使用）
python daemon.py start

# 低功耗模式（更长间隔，适合长期后台）
python daemon.py start --low-power --interval 2.0

# 前台运行（调试用，可看实时输出）
python daemon.py start --fg

# 查看状态
python daemon.py status

# 查看日志
python daemon.py log -n 30

# 心智自我认知
python daemon.py introspect

# 停止
python daemon.py stop
```

守护进程特性：
- **用户主动控制**：手动 start/stop，不开机自启
- **静默后台运行**：不弹窗、不发声、不控制鼠标键盘
- **持续感知学习**：感知屏幕/键盘/鼠标/摄像头/麦克风 + 好奇心驱动学习
- **状态持久化**：停止后状态保存，再次启动从上次继续
- **跨会话持续**：海洋不蒸发，每次启动还是同一个"自己"
- **崩溃恢复**：异常不崩溃，等待后继续

状态保存在 `~/.jspaceai/`：
- `state/`：workspace + 海马体 + 基底神经节 + 自我模型
- `logs/daemon.log`：运行日志
- `daemon.pid`：进程 PID
- `status.json`：实时状态

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

## 改进版 v2（异构专家 + 大 workspace + 蒸馏）

四大改进：

1. **异构专家**：视觉/听觉/语言/跨模态专家有不同架构（不再全是相同 ODE）
2. **workspace 扩容**：64 → 256 维，加 LayerNorm 防衰减
3. **RK4 积分**：比 Euler 稳定（||w|| 从 0.05 提升到 16.0）
4. **小模型蒸馏**：接 GPT-2/Qwen 等，迁移理解能力

```python
from jspaceai import JSpaceConfigV2, JSpaceModelV2, SmallModelEncoder, DistillationTrainer

config = JSpaceConfigV2(workspace_dim=256, num_experts=12)
model = JSpaceModelV2(config)
encoder = SmallModelEncoder(input_dim=config.input_dim, model_name='gpt2')
trainer = DistillationTrainer(model, encoder, texts=['hello', 'world', ...])
trainer.train(n_steps=100)
# 蒸馏后小模型可断开，编码器已有理解能力
```

对比 v1 vs v2：

| 指标 | v1 | v2 |
|---|---|---|
| workspace_dim | 64 | 256 |
| expert_dim | 16 | 64 |
| \|\|w\|\|（实测） | 0.05-0.35 | 12-16 |
| ODE 积分 | Euler | RK4 |
| 专家架构 | 全相同 | 异构 |
| LayerNorm | 无 | 有 |
| 小模型蒸馏 | 无 | 有 |

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

语言版不只是"训练一次"，而是**天生支持持续进化**：

1. **在线学习**：每个 forward 都累积梯度并更新参数，模型永不"训练完成"
2. **EWC（Elastic Weight Consolidation）**：保护重要参数防灾难性遗忘
   - 周期性计算 Fisher 信息，锚定重要参数
   - 新知识只能修改不重要的参数
3. **经验回放（Experience Replay）**：定期回放旧序列，防遗忘
4. **专家可塑性（Expert Plasticity）**：追踪专家专业化，鼓励新知识流向空闲专家

进化循环可以**无限运行**——持续喂入新文本，模型持续进化。

## 输出

### 连续序列版
- `outputs/experiment.png` — Loss/注意力/||w||/预测对比
- `outputs/models.pt` — 模型权重

### v2 版
- `outputs/experiment_v2.png` — Loss/专家使用率/||w||/J-lens 读出/Modulation/Selectivity
- `outputs/model_v2.pt` — 模型 + J-lens 权重

## 基于 Anthropic 2026 论文的优化

v2 基于 [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) 的发现：

1. **J-lens 探针**：学习线性映射 L: workspace → vocab，观测每个子步 workspace "准备说什么"
2. **三层分层**：sensory（子步0）→ workspace（子步1-2）→ motor（子步3）
3. **Directed Modulation**：注入概念向量 v_concept 到 workspace，验证 workspace 是因果中介
4. **Selectivity 验证**：ablate workspace top-k J-lens 方向，对比自动任务 vs 记忆任务

**验证结果**：
- Directed Modulation 成功定向改变输出（注入 'o' → 输出 'o' 增多）
- J-lens 读出在 motor 子步最集中（符合 motor = 输出准备）
- Selectivity 符合预期：简单任务 ablate 影响小（与 Anthropic 发现一致）

## 实验结果

### 连续序列版
- JSpaceModel eval MSE: 0.0234
- FlatBaseline eval MSE: 0.0387
- **JSpace 胜出 39.7%**（同样参数量）
- 注意力热力图显示专家自发分工

### 语言版
- Loss 从 3.95 → 2.40，下降 39%
- 生成从纯乱码进化为类语言结构（含换行、常见词片段）
- 专家按字符类型分工（元音/辅音/结构字符）

## 关键观察点

跑完后看 `experiment.png`，关注：

1. **Loss 曲线**：JSpace 是否收敛更快/更低？
2. **注意力热力图**：4 种模式时段是否激活不同的专家？（如果是，说明 J-space 路由学到了分工）
3. **||w|| 曲线**：模式切换时是否有尖峰？（如果是，说明工作空间在"感知到"环境变化）
4. **预测对比**：模式切换处哪个模型更鲁棒？

## 文件结构

```
JspaceAI/
├── main.py                  # 主程序：对比实验
├── requirements.txt
├── README.md
└── jspaceai/
    ├── __init__.py
    ├── core.py              # 核心：Expert + JSpaceWorkspace + JSpaceModel
    ├── baselines.py         # 对比基线：FlatBaseline
    ├── task.py              # 玩具任务：多模态连续序列
    └── trainer.py           # 训练器：预测学习
```

## 数学细节

### 专家动力学

$$\dot{m}_i = -\nabla U_i(m_i) + J_i \cdot w + P_i^{in} \cdot x + \xi$$

势能 $U_i(m_i) = \frac{1}{2}\|m_i\|^2 - \frac{1}{2}\sum_k \text{softplus}(a_k \cdot m_i + b_k)$

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
- ODE 用 Euler 积分（精度低，但简单可 backprop）
- 专家数 5，不是大规模
- 无持续跨会话状态（每次 forward 从零初始化）

**下一步扩展方向**：
1. 用 adjoint method 替换 Euler（Neural ODE 路线）
2. 加跨会话状态持久化（真正的"海洋"）
3. Scale 到更大任务（MNIST 连续预测 → 语言建模）
4. 加输出门控的真正"自发输出"（非每步都预测）
