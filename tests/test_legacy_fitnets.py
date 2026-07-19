from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from torch_fitnets.losses import (
    fitnet_teacher_weight,
    hint_mse_loss,
    legacy_kd_cross_entropy_loss,
)
from torch_fitnets.regressors import ConvHintRegressor


class LegacyFitNetsTest(unittest.TestCase):
    def test_maxout_regressor_matches_teacher_shape(self) -> None:
        regressor = ConvHintRegressor(
            student_channels=80,
            teacher_channels=192,
            student_hw=(8, 8),
            teacher_hw=(6, 6),
            num_pieces=2,
        )
        result = regressor(torch.randn(3, 80, 8, 8))
        self.assertEqual(tuple(result.shape), (3, 192, 6, 6))
        self.assertEqual(regressor.conv.out_channels, 384)

    def test_legacy_hint_reduction_matches_c01b_source(self) -> None:
        student = torch.zeros(2, 3, 4, 5)
        teacher = torch.ones_like(student)
        expected = 0.5 * (student - teacher).pow(2).sum(dim=2).mean()
        actual = hint_mse_loss(student, teacher, reduction="legacy_c01b")
        self.assertEqual(float(actual), float(expected))

    def test_legacy_kd_is_soft_cross_entropy_without_t_squared(self) -> None:
        student = torch.tensor([[0.2, -0.1, 0.5]])
        teacher = torch.tensor([[1.0, 0.0, -1.0]])
        temperature = 3.0
        expected = -(
            F.softmax(teacher / temperature, dim=1)
            * F.log_softmax(student / temperature, dim=1)
        ).sum(dim=1).mean()
        actual = legacy_kd_cross_entropy_loss(student, teacher, temperature)
        self.assertTrue(torch.allclose(actual, expected))

    def test_teacher_weight_schedule_matches_legacy_boundaries(self) -> None:
        self.assertEqual(fitnet_teacher_weight(1, 4.0), 4.0)
        self.assertEqual(fitnet_teacher_weight(5, 4.0), 4.0)
        self.assertLess(fitnet_teacher_weight(6, 4.0), 4.0)
        self.assertEqual(fitnet_teacher_weight(401, 4.0), 1.0)


if __name__ == "__main__":
    unittest.main()
