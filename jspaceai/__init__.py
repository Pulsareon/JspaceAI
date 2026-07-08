"""
JspaceAI —— 全局工作空间 + J-space 广播的智慧系统

模块：
    1. core.py: 核心架构（Expert + JSpaceWorkspace + JSpaceModel，含 RK4/LayerNorm/异构专家）
    2. language_model.py: 语言版 JSpace 模型
    3. jlens.py: J-lens 可解释性工具
    4. multimodal.py: 多模态（图像/音频/视频/文本）
    5. policy.py / runtime.py: 动作策略与 workspace 主循环
    6. events.py / memory.py: 可序列化事件与记忆接口
    7. training.py / continual.py: 可恢复训练与在线学习
"""
from .core import (
    Expert,
    JSpaceWorkspace,
    JSpaceModel,
    JSpaceConfig,
)
from .baselines import FlatBaseline
from .task import ContinuousSequenceTask
from .trainer import Trainer

from .language_data import CharTokenizer, CharDataset, load_shakespeare, load_chinese_corpus, load_textbook_corpus, load_corpus
from .language_model import (
    LanguageConfig,
    JSpaceLanguageModel,
    ExperienceReplay,
    EWCOptimizer,
    ExpertPlasticity,
)
from .jlens import (
    JLensConfig,
    JLensProbe,
    JLensSuite,
    WorkspaceAblator,
    DirectedModulation,
    CounterfactualReflection,
)
from .consensus import (
    ConsensusSlot,
    ConsensusSnapshot,
)
from .events import (
    WorkspaceEvent,
    ActionEvent,
)
from .memory import (
    MemoryRecord,
    MemoryStore,
    InMemoryVectorMemoryStore,
    Hippocampus,
)
from .multimodal import (
    MultimodalConfig,
    MultimodalJSpaceModel,
    VisualEncoder, VisualDecoder,
    AudioEncoder, AudioDecoder,
    TextEncoder, TextDecoder,
)
from .continual import (
    OnlineLanguageLearner,
)
from .training import (
    LanguageTrainingConfig,
    LanguageTrainingSession,
    TokenBatchSampler,
    expert_integration_mode,
    save_language_checkpoint,
)
from .policy import (
    ACTION_LABELS,
    compose_action_params,
    ReflexRule,
    ActionGate,
    ActionValueModel,
    MotorController,
    ActionDecision,
    ActionPolicy,
)
from .realtime import (
    Frame,
    CameraStream,
    MicrophoneStream,
    AudioPlayer,
    MultimodalStream,
    SensoryMotorLoop,
)
from .desktop import (
    InputEvent,
    ScreenCapture,
    KeyboardMonitor,
    MouseMonitor,
    DesktopStream,
    FullSensoryStream,
)
from .platform import (
    PlatformInfo, PLATFORM,
    get_screen_size,
    check_camera_permission, check_microphone_permission,
    check_input_monitoring_permission, print_permission_guide,
)
from .embodied import (
    MouseActuator, KeyboardActuator, AudioActuator, ScreenActuator,
    Cerebellum, CentralNervousSystem, BasalGanglia,
    EmbodiedAgent,
)
from .runtime import (
    CuriosityDrive, RuntimeStateStore, PersistentState, SelfModel, MetaLearner,
    WorkspaceRuntime, AutonomousMind,
)
from .modules import (
    ExternalModule, SmallModelModule, KnowledgeBaseModule,
    ToolModule, ModuleDock,
)
from .evolution import EvolutionTrainer

__all__ = [
    # 核心架构
    "Expert", "JSpaceWorkspace", "JSpaceModel", "JSpaceConfig",
    # 对比基线
    "FlatBaseline",
    # 连续序列任务
    "ContinuousSequenceTask", "Trainer",
    # 语言建模
    "CharTokenizer", "CharDataset", "load_shakespeare",
    "load_chinese_corpus", "load_textbook_corpus", "load_corpus",
    "LanguageConfig", "JSpaceLanguageModel",
    "ExperienceReplay", "EWCOptimizer", "ExpertPlasticity",
    # J-lens 可解释性
    "JLensConfig", "JLensProbe", "JLensSuite",
    "WorkspaceAblator", "DirectedModulation", "CounterfactualReflection",
    "ConsensusSlot", "ConsensusSnapshot",
    "WorkspaceEvent", "ActionEvent",
    "MemoryRecord", "MemoryStore", "InMemoryVectorMemoryStore",
    # 多模态
    "MultimodalConfig", "MultimodalJSpaceModel",
    "VisualEncoder", "VisualDecoder",
    "AudioEncoder", "AudioDecoder",
    "TextEncoder", "TextDecoder",
    # 动作策略
    "ACTION_LABELS", "compose_action_params", "ReflexRule",
    "ActionGate", "ActionValueModel", "MotorController",
    "ActionDecision", "ActionPolicy",
    # 实时 I/O
    "Frame", "CameraStream", "MicrophoneStream", "AudioPlayer",
    "MultimodalStream", "SensoryMotorLoop",
    # 桌面 I/O
    "InputEvent", "ScreenCapture", "KeyboardMonitor", "MouseMonitor",
    "DesktopStream", "FullSensoryStream",
    # 平台抽象
    "PlatformInfo", "PLATFORM", "get_screen_size",
    "check_camera_permission", "check_microphone_permission",
    "check_input_monitoring_permission", "print_permission_guide",
    # 具身 Agent（输出执行器 + 神经系统）
    "MouseActuator", "KeyboardActuator", "AudioActuator", "ScreenActuator",
    "Cerebellum", "CentralNervousSystem", "Hippocampus", "BasalGanglia",
    "EmbodiedAgent",
    # 自主心智（最重要的能力）
    "CuriosityDrive", "RuntimeStateStore", "PersistentState",
    "SelfModel", "MetaLearner", "WorkspaceRuntime", "AutonomousMind",
    # 外挂模块系统（可热插拔）
    "ExternalModule", "SmallModelModule", "KnowledgeBaseModule",
    "ToolModule", "ModuleDock",
    # 自主进化
    "EvolutionTrainer",
    "OnlineLanguageLearner",
    "LanguageTrainingConfig", "LanguageTrainingSession",
    "TokenBatchSampler", "expert_integration_mode",
    "save_language_checkpoint",
]
