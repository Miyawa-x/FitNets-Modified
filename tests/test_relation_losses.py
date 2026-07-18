from __future__ import annotations

import unittest

import torch

from torch_fitnets.losses import relation_distance_loss, relation_similarity_loss


class RelationLossTest(unittest.TestCase):
    def test_equivalent_geometry_across_dimensions_has_zero_loss(self) -> None:
        teacher = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        student = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        )

        self.assertLess(float(relation_distance_loss(student, teacher)), 1e-6)
        self.assertLess(float(relation_similarity_loss(student, teacher)), 1e-6)

    def test_relation_losses_backpropagate_to_student_only(self) -> None:
        torch.manual_seed(7)
        student = torch.randn(4, 3, 2, 2, requires_grad=True)
        teacher = torch.randn(4, 5, 3, 3, requires_grad=True)

        loss = relation_distance_loss(student, teacher)
        loss = loss + relation_similarity_loss(student, teacher)
        loss.backward()

        self.assertIsNotNone(student.grad)
        self.assertTrue(bool(torch.isfinite(student.grad).all()))
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)


if __name__ == "__main__":
    unittest.main()
