import unittest

import torch

from jspaceai import (
    CharTokenizer,
    JSpaceConfig,
    JSpaceLanguageModel,
    JSpaceModel,
    LanguageConfig,
    MultimodalConfig,
    MultimodalJSpaceModel,
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
        self.assertEqual(len(info["w_trajectory"]), 2)

    def test_tokenizer_unknown_maps_to_zero(self):
        tokenizer = CharTokenizer.from_text("abc")

        self.assertEqual(tokenizer.encode("?"), [0])
        self.assertEqual(tokenizer.decode([0]), "<unk>")


if __name__ == "__main__":
    unittest.main()
