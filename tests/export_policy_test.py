"""Run with: python -m unittest discover -s tests -p '*_test.py'."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import export_policy


class DefaultPoseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.checkpoint = self.run / "checkpoints" / "000000001"
        self.checkpoint.mkdir(parents=True)
        self.pose = np.array([0, -0.405, 0.84, -0.42, 0, 0.39, -0.84, 0.42], dtype=np.float32)

    def write_xml(self, ctrl=None, *, name="basic", path=None):
        path = path or self.run / "model.xml"
        path.parent.mkdir(parents=True, exist_ok=True)
        attributes = f'name="{name}" qpos="99 98 97"'
        if ctrl is not None:
            attributes += f' ctrl="{ctrl}"'
        path.write_text(f"<mujoco><keyframe><key {attributes}/></keyframe></mujoco>")
        return path

    def bundle(self):
        bundle = self.run / "diagnostics" / "final"
        model = self.write_xml(
            " ".join(map(str, self.pose)), path=bundle / "assets" / "robot.xml"
        )
        (bundle / "metadata.json").write_text(json.dumps({
            "model_source": {"path": "/remote/training/robot.xml"}
        }))
        # Another XML in the bundle must not determine the exported pose.
        self.write_xml("0 0 0 0 0 0 0 0", path=bundle / "assets" / "other.xml")
        return model

    def test_saved_model_targets_in_both_checkpoint_layouts(self):
        model = self.bundle()
        for checkpoint in (self.checkpoint, self.run / "000000001"):
            with self.subTest(checkpoint=checkpoint):
                pose, source = export_policy._load_default_pose(checkpoint, 8)
                np.testing.assert_array_equal(pose, self.pose)
                self.assertIn(str(model), source)

    def test_explicit_model_overrides_bundle(self):
        self.bundle()
        model = self.write_xml("1 2 3 4 5 6 7 8")
        pose, _ = export_policy._load_default_pose(self.checkpoint, 8, model)
        np.testing.assert_array_equal(pose, np.arange(1, 9))

    def test_missing_assets_use_zero(self):
        model = self.bundle()
        model.unlink()
        pose, source = export_policy._load_default_pose(self.checkpoint, 8)
        np.testing.assert_array_equal(pose, np.zeros(8))
        self.assertIn("using zero pose", source)

    def test_missing_model_or_keyframe_ctrl_use_zero(self):
        models = [None, self.write_xml("1 2 3 4 5 6 7 8", name="other")]
        for model in models:
            pose, source = export_policy._load_default_pose(self.checkpoint, 8, model)
            np.testing.assert_array_equal(pose, np.zeros(8))
            self.assertIn("using zero pose", source)
        model = self.write_xml()  # qpos must never substitute for ctrl.
        pose, _ = export_policy._load_default_pose(self.checkpoint, 8, model)
        np.testing.assert_array_equal(pose, np.zeros(8))

    def test_invalid_targets_fail(self):
        for ctrl in ("1 2", "", "nan 0 0 0 0 0 0 0", "inf 0 0 0 0 0 0 0", "junk"):
            with self.subTest(ctrl=ctrl), self.assertRaises(ValueError):
                export_policy._load_default_pose(self.checkpoint, 8, self.write_xml(ctrl))

    def test_explicit_missing_model_fails(self):
        with self.assertRaises(FileNotFoundError):
            export_policy._load_default_pose(self.checkpoint, 8, self.run / "missing.xml")

    def test_header_contains_pose_without_changing_policy(self):
        self.bundle()
        (self.checkpoint.parent / "config.json").write_text(json.dumps({
            "command_history_length": 2, "ctrl_dt": 0.02,
            "step_frequency": 1.2, "gyro_obs_scale": 0.25,
            "target_velocity": 0.068, "action_scale": 0.4,
        }))
        (self.checkpoint / "ppo_network_config.json").write_text(json.dumps({
            "observation_size": {"shape": [25]}, "action_size": 8,
            "normalize_observations": True,
            "network_factory_kwargs": {
                "policy_hidden_layer_sizes": [], "activation": "silu",
                "distribution_type": "tanh_normal",
            },
        }))
        restored = [
            {"mean": np.zeros(25, dtype=np.float32), "std": np.ones(25, dtype=np.float32)},
            {"params": {"hidden_0": {
                "kernel": np.zeros((25, 16), dtype=np.float32),
                "bias": np.zeros(16, dtype=np.float32),
            }}},
        ]
        with mock.patch.object(export_policy, "_restore_checkpoint", return_value=restored):
            header, count, error = export_policy._build_header(self.checkpoint)
        self.assertIn(export_policy._format_array("POLICY_DEFAULT_POSE_RAD", self.pose), header)
        self.assertEqual(count, 416)
        self.assertEqual(error, 0)


if __name__ == "__main__":
    unittest.main()
