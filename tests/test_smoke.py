import unittest
from pathlib import Path
import tempfile

import torch

from jspaceai import (
    ActionEvent,
    ActionPolicy,
    ChatBatchSampler,
    CharTokenizer,
    ConsensusSnapshot,
    InMemoryVectorMemoryStore,
    JSpaceConfig,
    JSpaceLanguageModel,
    JSpaceModel,
    LanguageConfig,
    LanguageTrainingConfig,
    LanguageTrainingSession,
    MultimodalConfig,
    MultimodalJSpaceModel,
    OnlineLanguageLearner,
    WorkspaceEvent,
    WorkspaceRuntime,
    build_child_chat_corpus,
    compose_action_params,
    extract_child_reply,
    format_child_prompt,
    load_child_dialog_examples,
    lookup_child_reply,
)
from main_chat import generate_response


class SmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_core_forward_shapes(self):
        config = JSpaceConfig(
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=3,
            ode_steps=2,
            noise_std=0.0,
        )
        model = JSpaceModel(config)
        xs = torch.randn(2, 5, 8)

        preds, info = model(xs, record_trajectory=True)

        self.assertEqual(tuple(preds.shape), (2, 5, 8))
        self.assertEqual(tuple(info["alpha"].shape), (2, 5, 3))
        self.assertEqual(tuple(info["w_trajectory"].shape), (2, 5, 2, 16))

    def test_language_fast_generation_preserves_eval_and_rk4(self):
        tokenizer = CharTokenizer.from_text("学而时习之")
        config = LanguageConfig(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=3,
            ode_steps=1,
            noise_std=0.0,
        )
        model = JSpaceLanguageModel(config)
        model.eval()

        response = generate_response(
            model,
            tokenizer,
            "学",
            n_new=3,
            temperature=1.0,
            top_k=2,
            fast=True,
        )

        self.assertEqual(len(response), 3)
        self.assertFalse(model.training)
        self.assertTrue(all(expert.use_rk4 for expert in model.experts))

    def test_language_generate_preserves_training_mode(self):
        tokenizer = CharTokenizer.from_text("abcabc")
        config = LanguageConfig(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=3,
            ode_steps=1,
            noise_std=0.0,
        )
        model = JSpaceLanguageModel(config)
        model.eval()
        model.generate([1], n_new=1)
        self.assertFalse(model.training)
        model.train()
        model.generate([1], n_new=1)
        self.assertTrue(model.training)

    def test_multimodal_single_step_records_trajectory(self):
        config = MultimodalConfig(
            vocab_size=20,
            embed_dim=8,
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=4,
            ode_steps=2,
            noise_std=0.0,
            audio_frame_size=128,
        )
        model = MultimodalJSpaceModel(config)
        token = torch.tensor([1])

        outputs, info = model.forward_multimodal(
            "text",
            token,
            record_trajectory=True,
        )

        self.assertEqual(tuple(outputs["w"].shape), (1, 16))
        self.assertIn("alpha", info)
        self.assertEqual(len(info["w_trajectory"]), 2)

    def test_tokenizer_unknown_maps_to_zero(self):
        tokenizer = CharTokenizer.from_text("abc")

        self.assertEqual(tokenizer.encode("?"), [0])
        self.assertEqual(tokenizer.decode([0]), "<unk>")

    def test_consensus_snapshot_extracts_primary_slot(self):
        w = torch.ones(1, 4)
        alpha = torch.tensor([[0.1, 0.7, 0.2]])

        snapshot = ConsensusSnapshot.from_workspace(
            w,
            alpha,
            modality="text",
            labels=["vision", "language", "memory"],
        )

        self.assertEqual(snapshot.primary_slot().label, "language")
        self.assertGreater(snapshot.confidence, 0.0)

    def test_online_language_learner_tracks_session_state(self):
        tokenizer = CharTokenizer.from_text("学而时习之学而时习之")
        config = LanguageConfig(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=3,
            ode_steps=1,
            noise_std=0.0,
        )
        model = JSpaceLanguageModel(config)
        learner = OnlineLanguageLearner(
            model,
            config,
            tokenizer,
            seq_len=8,
            replay_batch_size=1,
            consolidate_every=2,
        )

        first = learner.learn_text("学而时习之")
        second = learner.learn_text("学而时习之")

        self.assertEqual(first["step"], 1)
        self.assertEqual(second["step"], 2)
        self.assertGreaterEqual(second["replay_loss"], 0.0)

    def test_language_training_session_saves_checkpoint(self):
        tokenizer = CharTokenizer.from_text("学而时习之学而时习之")
        config = LanguageConfig(
            vocab_size=tokenizer.vocab_size,
            embed_dim=8,
            input_dim=8,
            workspace_dim=16,
            expert_dim=8,
            num_experts=3,
            ode_steps=1,
            noise_std=0.0,
        )
        model = JSpaceLanguageModel(config)
        train_config = LanguageTrainingConfig(
            seq_len=8,
            batch_size=2,
            lr=1e-2,
            replay_batch_size=1,
            validate_every=1,
            validate_batches=1,
            save_every=1,
            consolidate_every=0,
        )
        session = LanguageTrainingSession(
            model, config, tokenizer, train_config, device="cpu",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "model.pt"
            history = session.fit_text(
                "学而时习之学而时习之",
                max_steps=2,
                checkpoint_path=ckpt_path,
            )
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.assertEqual(len(history), 2)
        self.assertIn("trainer", ckpt)
        self.assertEqual(ckpt["trainer"]["global_step"], 2)

    def test_child_chat_sampler_masks_answer_tokens(self):
        examples = load_child_dialog_examples(repeats=1)
        tokenizer = CharTokenizer.from_text(build_child_chat_corpus(repeats=1))
        sampler = ChatBatchSampler(
            examples[:2],
            tokenizer,
            seq_len=48,
            train_fraction=0.5,
        )

        token_seq, loss_mask = sampler.sample(2)
        prompt_len = len(tokenizer.encode(format_child_prompt(examples[0].user)))

        self.assertEqual(tuple(token_seq.shape), (2, 48))
        self.assertEqual(tuple(loss_mask.shape), (2, 47))
        self.assertGreater(loss_mask.sum().item(), 0)
        self.assertTrue(all(v == 0 for v in loss_mask[0, :prompt_len - 1].tolist()))
        self.assertEqual(extract_child_reply("问：你好\n答：你好呀。\n问：再见"), "你好呀。")
        self.assertEqual(lookup_child_reply("你好"), "你好呀。")

    def test_compose_action_params_respects_discrete_mode(self):
        raw = torch.tensor([0.5, -0.25, 0.3, -0.4, 0.8]).numpy()

        move = compose_action_params(1, raw)
        left_click = compose_action_params(2, raw)

        self.assertEqual(move.tolist(), [0.5, -0.25, 0.0, 0.0, 0.0])
        self.assertEqual(left_click.tolist(), [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_action_policy_decides_from_workspace(self):
        policy = ActionPolicy(workspace_dim=4, exploration=0.0)
        policy.value_model.action_weights[4, 0] = 2.0
        w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        decision = policy.decide(w)

        self.assertEqual(decision.action_idx, 4)
        self.assertEqual(decision.action_name, "scroll")

    def test_workspace_event_and_memory_store_round_trip(self):
        event = WorkspaceEvent.from_tensor(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            modality="text",
            step=7,
            context={"source": "test"},
        )
        action = ActionEvent.from_action_info({
            "action_idx": 1,
            "action_name": "mouse_move",
            "action_params": [0.1, 0.0, 0.0, 0.0, 0.0],
            "executed": True,
            "risk": 0.0,
        }, step=7)
        memory = InMemoryVectorMemoryStore(capacity=4, workspace_dim=4)

        memory.put(event)
        results = memory.query(torch.tensor([[0.9, 0.0, 0.0, 0.0]]), top_k=1)

        self.assertEqual(event.to_dict()["step"], 7)
        self.assertEqual(action.to_dict()["action_name"], "mouse_move")
        self.assertEqual(results[0].event.modality, "text")
        self.assertGreater(results[0].similarity, 0.9)

    def test_workspace_runtime_runs_minimal_loop(self):
        class DummySenses:
            def start(self):
                return None

            def stop(self):
                return None

        class DummyAudio:
            def stop(self):
                return None

        class DummyMemory:
            def __init__(self):
                self.items = []

            def store(self, w, context=None):
                self.items.append((w, context or {}))

            def size(self):
                return len(self.items)

        class DummyConfig:
            workspace_dim = 4

        class DummyAgent:
            def __init__(self):
                self.config = DummyConfig()
                self.state = {
                    "w": torch.zeros(1, 4),
                    "m": [torch.zeros(1, 2)],
                }
                self.senses = DummySenses()
                self.audio_actuator = DummyAudio()
                self.hippocampus = DummyMemory()
                self.last_consensus = None
                self.learned = []

            def perceive(self):
                return {}

            def think(self, sensory_data):
                del sensory_data
                self.state["w"] = self.state["w"] + 0.25
                return self.state["w"], "idle"

            def decide_and_act(self, w, modality):
                del w, modality
                return {
                    "action_idx": 1,
                    "action_name": "mouse_move",
                    "action_params": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "raw_action_params": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "executed": False,
                    "action_strength": 0.0,
                    "risk": 0.0,
                }

            def remember(self, w, context):
                self.hippocampus.store(w[0].numpy(), context)

            def learn(self, w, action_idx, reward=0.0):
                del w
                self.learned.append((action_idx, reward))

        agent = DummyAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = WorkspaceRuntime(agent, save_dir=Path(tmpdir), device="cpu")
            info = runtime.step()

        self.assertEqual(info["step"], 1)
        self.assertEqual(info["modality"], "idle")
        self.assertEqual(info["memory_count"], 1)
        self.assertEqual(agent.learned[0][0], 1)
        self.assertEqual(info["workspace_event"]["step"], 1)
        self.assertEqual(info["action_event"]["action_name"], "mouse_move")


if __name__ == "__main__":
    unittest.main()
