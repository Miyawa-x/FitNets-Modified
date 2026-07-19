from __future__ import annotations

import unittest

from torch_fitnets.models import build_model


class ModelDropoutTest(unittest.TestCase):
    def test_cifar_teacher_uses_maxout_dropout_recipe(self) -> None:
        teacher = build_model(
            "maxout_cifar_teacher",
            input_channels=3,
            num_classes=100,
            image_size=32,
        )

        self.assertEqual(teacher.input_dropout.p, 0.2)
        self.assertEqual(teacher.hidden_dropout.p, 0.5)
        self.assertEqual(teacher.fc_dropout.p, 0.5)

    def test_dropout_override_does_not_change_checkpoint_parameter_keys(self) -> None:
        default_teacher = build_model(
            "maxout_cifar_teacher",
            input_channels=3,
            num_classes=100,
            image_size=32,
        )
        no_dropout_teacher = build_model(
            "maxout_cifar_teacher",
            input_channels=3,
            num_classes=100,
            image_size=32,
            input_dropout=0.0,
            hidden_dropout=0.0,
        )

        self.assertEqual(no_dropout_teacher.input_dropout.p, 0.0)
        self.assertEqual(no_dropout_teacher.hidden_dropout.p, 0.0)
        self.assertEqual(
            set(default_teacher.state_dict()),
            set(no_dropout_teacher.state_dict()),
        )

    def test_cifar_student_keeps_dropout_disabled(self) -> None:
        student = build_model(
            "fitnet19_cifar_student",
            input_channels=3,
            num_classes=100,
            image_size=32,
        )

        self.assertEqual(student.input_dropout.p, 0.0)
        self.assertEqual(student.hidden_dropout.p, 0.0)


if __name__ == "__main__":
    unittest.main()
