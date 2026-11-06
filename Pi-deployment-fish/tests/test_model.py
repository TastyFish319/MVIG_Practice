import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures import checkpoint
from pi_deploy.model.spec import resolve_spec, checkpoint_directory
from pi_deploy.model.features import build_state, decode_actions, letterbox_rgb
from pi_deploy.control.history import StateHistory
from pi_deploy.inference.gripper import GripperFilter
from pi_deploy.inference.options import gripper_options


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_right_state_order_and_absolute_action(self):
        s = resolve_spec(checkpoint(self.root / 'right'))
        h = StateHistory()
        robot = np.arange(14, dtype=float) / 10
        h.add(2, robot)
        state, anchor, info = build_state(s, h, 2, 0)
        np.testing.assert_allclose(state, robot[list(range(7, 14)) + list(range(7))])
        self.assertFalse(info['velocity_padded'])
        raw = np.full((50, 7), 0.3)
        out = decode_actions(s, raw, anchor, robot)
        np.testing.assert_allclose(out[:, 7:], raw)  # no second delta addition
        np.testing.assert_allclose(out[:, :7], np.tile(robot[:7], (50, 1)))

    def test_both_actions_map_by_name(self):
        s = resolve_spec(checkpoint(self.root, both=True))
        raw = np.tile(np.arange(14) * 0.03, (50, 1))
        out = decode_actions(s, raw, np.ones(14), np.zeros(14))
        np.testing.assert_allclose(out[:, :7], raw[:, 7:])
        np.testing.assert_allclose(out[:, 7:], raw[:, :7])

    def test_velocity_fit_and_insufficient_samples_padding(self):
        s = resolve_spec(checkpoint(self.root, velocity=True))
        h = StateHistory()
        h.add(1.0, np.zeros(14))
        self.assertTrue(build_state(s, h, 1, 1)[2]['velocity_padded'])
        h.add(1.01, np.full(14, 0.02))
        st, _, info = build_state(s, h, 1.01, 1)
        np.testing.assert_allclose(st[14:], 2, atol=1e-5)
        self.assertFalse(info['velocity_padded'])  # not ACT's first-two-zero rule
        for i in range(2, 31):
            h.add(1 + i / 100, np.full(14, i / 50))
        np.testing.assert_allclose(build_state(s, h, 1.3, 1.29)[0][14:], 2, atol=1e-5)

    def test_image_is_deterministic_rgb_letterbox(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[:] = [250, 10, 0]
        a = letterbox_rgb(rgb, 224, 224, None)
        self.assertEqual(a.shape, (224, 224, 3))
        self.assertTrue((a[:28] == 0).all())
        self.assertTrue((a[28:196, :, 0] == 250).all())
        np.testing.assert_array_equal(a, letterbox_rgb(rgb, 224, 224, None))

    def test_floor_off_still_clips_and_on_retains_parameters(self):
        contract = {'gripper_filter': dict(floor=0.457, knee=0.587, enabled=True)}
        off = GripperFilter(**gripper_options(contract, 'off'))
        on = GripperFilter(**gripper_options(contract, 'on'))
        np.testing.assert_allclose(off([-1, 0, 1.2]), [0, 0, 1])
        self.assertGreater(float(on(0)), .457)
        self.assertEqual(on.floor, .457)
        self.assertEqual(gripper_options(contract, 'checkpoint'), contract['gripper_filter'])

    def test_missing_processor_state_rejected(self):
        checkpoint(self.root)
        (self.root / 'pre.safetensors').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            checkpoint_directory(self.root)

    def test_wrong_contract_or_order_rejected(self):
        checkpoint(self.root)
        p = self.root / 'pitrain_contract.json'
        c = json.loads(p.read_text())
        c['action_names'] = c['state_names'][7:]
        p.write_text(json.dumps(c))
        with self.assertRaises(ValueError):
            resolve_spec(self.root)

    def test_nonfinite_actions_rejected(self):
        s = resolve_spec(checkpoint(self.root))
        with self.assertRaises(ValueError):
            decode_actions(s, np.full((50, 7), np.nan), np.zeros(14), np.zeros(14))

    def test_same_shape_config_contract_name_mismatch_rejected(self):
        checkpoint(self.root)
        p = self.root / 'config.json'
        cfg = json.loads(p.read_text())
        cfg['action_feature_names'] = [f'left_joint{i}' for i in range(6)] + ['left_gripper']
        p.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            resolve_spec(self.root)

    def test_actual_server_contract_metadata(self):
        source = Path(__file__).resolve().parents[1]/'docs/reference-contracts.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        self.assertTrue(data['metadata_only'])
        for item in data['items']:
            folder = self.root/item['run']; folder.mkdir()
            for name, body in item['files'].items():
                (folder/name).write_text(json.dumps(body), encoding='utf-8')
            # Metadata acceptance only. No real network weights are present.
            sf = set(['model.safetensors'])
            for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
                sf.update(s['state_file'] for s in item['files'][name]['steps'] if s.get('state_file'))
            for name in sf:
                (folder/name).write_bytes((2).to_bytes(8, 'little')+b'{}')
            spec = resolve_spec(folder)
            self.assertEqual(spec.horizon, 50)
            self.assertEqual(spec.state_dim, 14)
            self.assertEqual(spec.action_dim, 7 if item['run'].startswith('blue_block') else 14)

    def test_absolute_workspace_range_uses_input_state_not_delta_actions(self):
        checkpoint(self.root)
        lo = np.arange(14, dtype=np.float32) * .01; hi = lo + 1
        header = {'observation.state.min': dict(dtype='F32', shape=[14], data_offsets=[0,56]),
                  'observation.state.max': dict(dtype='F32', shape=[14], data_offsets=[56,112])}
        body = json.dumps(header).encode()
        (self.root/'pre.safetensors').write_bytes(struct.pack('<Q',len(body))+body+lo.tobytes()+hi.tobytes())
        spec = resolve_spec(self.root)
        np.testing.assert_allclose(spec.state_min[:7], lo[7:])
        np.testing.assert_allclose(spec.state_max[7:], hi[:7])


if __name__ == '__main__':
    unittest.main()
