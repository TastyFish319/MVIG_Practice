import socket
import struct
import json
import time
import unittest
import numpy as np
from pi_deploy.config import vec14, ConfigError, HARDWARE_YAML
from pi_deploy.hw.follower.config import load_teleoperation_config
from pi_deploy.hw.follower.runtime import WorkerControlRuntime
from pi_deploy.hw.follower.endpoints import FakeFollowerEndpoint
from pi_deploy.ipc import recv_msg, MAGIC
from pi_deploy.hw.arms import FollowerArms


class InfrastructureTests(unittest.TestCase):
    def test_single_right_arm_command_uses_right_slots_only(self):
        arm = FollowerArms.__new__(FollowerArms)
        arm.sides = ('right',); arm.session = 'test'; arm.seq = 0
        sent = []
        arm._both = lambda method, args: sent.append(args)
        target = np.arange(14, dtype=float)
        arm._send(target, 'ACTIVE')
        self.assertEqual(set(sent[0]), {'right'})
        np.testing.assert_array_equal(sent[0]['right'][0]['joint_position'], target[7:])

    def test_nonfinite_pose_rejected(self):
        bad = [0.] * 14
        bad[0] = float('inf')
        with self.assertRaises(ConfigError):
            vec14(bad, 'pose')

    def test_ipc_rejects_object_array(self):
        a, b = socket.socketpair()
        try:
            head = json.dumps({'arrays': [{'name': 'state', 'dtype': '|O', 'shape': [1]}]}).encode()
            a.sendall(MAGIC + struct.pack('!I', len(head)) + head)
            with self.assertRaises(ConnectionError):
                recv_msg(b)
        finally:
            a.close(); b.close()

    def test_worker_watchdog_and_recovery_with_fake_endpoint(self):
        cfg = load_teleoperation_config(HARDWARE_YAML)
        runtime = WorkerControlRuntime(FakeFollowerEndpoint(), role='follower', side='right', control=cfg.control)
        try:
            self.assertTrue(runtime.start(2))
            runtime.begin_control_session('test')
            sample = runtime.get_sample()['state']
            runtime.submit_control_frame(dict(session_id='test', sequence=1, issued_at_ns=time.monotonic_ns(),
                                              phase='ALIGNING', joint_position=sample))
            end = time.monotonic() + 1
            while runtime.get_status()['last_applied_sequence'] < 1 and time.monotonic() < end:
                time.sleep(.005)
            runtime.set_phase('test', 'ACTIVE')
            # Deliberately stop sending frames; the independent worker must fault-hold.
            end = time.monotonic() + 1
            while runtime.get_status()['phase'] != 'FAULT' and time.monotonic() < end:
                time.sleep(.01)
            self.assertEqual(runtime.get_status()['phase'], 'FAULT')
            self.assertTrue(np.isfinite(runtime.get_sample()['state']).all())
            runtime.reset_fault()
            self.assertEqual(runtime.get_status()['phase'], 'SAFE_HOLD')
        finally:
            self.assertTrue(runtime.close())
