"""Mock TCP + actual threaded engine. No robot libraries or torch are imported."""
import argparse
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures import checkpoint
from pi_deploy.config import load_deploy
from pi_deploy.app.engine import Engine, EngineOptions
from pi_deploy.app.main import free_port
from pi_deploy.hw.arms import SimArms, ArmFault
from pi_deploy.hw.cameras import CameraRig
from pi_deploy.ipc import PolicyClient
from pi_deploy.model.features import dummy_inputs
from pi_deploy.model.spec import resolve_spec
from pi_deploy.server.policy_server import Server


def wait_for(predicate, label, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f'timeout: {label}')
        time.sleep(.01)


class IntegrationTests(unittest.TestCase):
    def test_mock_tcp_engine_lifecycle_and_inflight_estop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = resolve_spec(checkpoint(root / 'ckpt', both=True))
            port = free_port(28890)
            args = argparse.Namespace(checkpoint=spec.checkpoint, port=port, device='cpu', backend='mock',
                                      warmup=2, keep_warm_s=0, exit_with_parent=False,
                                      model_overrides='{}', tokenizer='', gripper_floor='off', seed=0)
            server = Server(args)
            thread = threading.Thread(target=server.serve, daemon=True)
            thread.start()
            client = PolicyClient('127.0.0.1', port, timeout_s=2)
            engine = cams = None
            try:
                wait_for(lambda: server.status == 'ready', 'server ready')
                health = client.health()
                self.assertEqual(health['spec']['fingerprint'], spec.fingerprint)
                state, images = dummy_inputs(spec)
                out, reply = client.predict(state, images)
                self.assertEqual(out.shape, (50, 14))
                self.assertIsInstance(reply['request_id'], int)
                dep = load_deploy()
                motion = dict(dep['motion'], min_duration_s=.1)
                zero = np.asarray(dep['poses']['zero'])
                start = zero.copy(); start[0] = .025; start[7] = -.025
                cams = CameraRig.fake(['left_wrist', 'right_wrist'], 640, 480, 30)
                cams.wait_ready()
                opts = EngineOptions(mode='real', rate_hz=100, motion=motion, poses=dep['poses'],
                                     safety=dep['safety'], policy=dep['policy'],
                                     preset=dict(dep['execution']['presets']['chunk_auto'], exec_steps=15),
                                     recording=dict(enabled=True, save_images=False), max_frame_age_s=.2,
                                     start_pose=start, max_episode_s=3, session_dir=root, meta={'test': True})
                engine = Engine(opts, spec, SimArms(zero, tau_s=.02), cams, client)
                engine.start()
                wait_for(lambda: engine.warmup_ok, 'live warmup')
                engine.command('next')
                wait_for(lambda: engine.phase == 'READY', 'initialise')
                engine.command('next')
                wait_for(lambda: engine.n_infer >= 2, 'inference')
                engine.command('pause')
                wait_for(lambda: engine.phase == 'PAUSED', 'pause')
                old_generation = engine.episode_id
                engine.command('next')
                wait_for(lambda: engine.phase == 'RUNNING', 'resume')
                self.assertGreater(engine.episode_id, old_generation)
                engine.command('estop')
                wait_for(lambda: engine.phase == 'ESTOP', 'estop')
                time.sleep(.1)
                self.assertEqual(engine.phase, 'ESTOP')
                engine.command('zero')
                wait_for(lambda: engine.phase == 'ZERO', 'zero')
                engine.command('next')
                wait_for(lambda: engine.phase == 'READY', 'next episode init')
                engine.command('next')
                wait_for(lambda: engine.phase == 'RUNNING', 'next episode')
                engine.command('finish')
                wait_for(lambda: engine.phase == 'ZERO', 'finish return')
                engine.command('release')
                self.assertTrue(engine.wait_done(8))
                self.assertEqual(engine.phase, 'RELEASED')
                for save in engine._save_threads:
                    save.join(5)
                self.assertEqual(len(list((root / 'episodes').glob('*.npz'))), 2)
            finally:
                if engine:
                    engine.stop.set()
                    for worker in engine._threads:
                        worker.join(3)
                if cams:
                    cams.close()
                client.shutdown()
                server.stop.set()
                thread.join(3)


if __name__ == '__main__':
    unittest.main()
