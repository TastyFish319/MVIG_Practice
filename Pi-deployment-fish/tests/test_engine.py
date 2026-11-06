import copy
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tests.fixtures import checkpoint
from pi_deploy.app.engine import Engine, EngineOptions
from pi_deploy.config import load_deploy
from pi_deploy.control.scheduler import Plan
from pi_deploy.control.trajectory import Route
from pi_deploy.hw.arms import SimArms
from pi_deploy.model.spec import resolve_spec


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name)
        self.spec = resolve_spec(checkpoint(p / 'ckpt'))
        self.d = load_deploy()
        self.d['recording']['enabled'] = False
        preset = dict(self.d['execution']['presets']['chunk_auto'], exec_steps=15)
        o = EngineOptions(mode='real', rate_hz=100, motion=self.d['motion'], poses=self.d['poses'],
                          safety=self.d['safety'], policy=self.d['policy'], preset=preset,
                          recording=self.d['recording'], max_frame_age_s=.2,
                          start_pose=np.zeros(14), max_episode_s=80., session_dir=p, meta={})
        e = self.e = Engine(o, self.spec, SimArms(np.zeros(14)), None, None)
        e.state = np.zeros(14); e.cmd = np.zeros(14); e.hold = np.zeros(14)
        e.ep_hold = np.zeros(14); e.safety.reset(e.cmd); e.phase = 'READY'
        e.plan_wait_since = time.monotonic()
        e.cams = SimpleNamespace(grab=lambda *args: ({}, {}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_live_warmup_gates_execution(self):
        self.e._handle('next', None, time.monotonic())
        self.assertEqual(self.e.phase, 'READY')

    def test_idle_release_routes_to_zero(self):
        e = self.e; e.phase = 'IDLE'; e.state[:6] = .2; e.cmd = e.state.copy()
        e._handle('release', None, time.monotonic())
        self.assertEqual(e.phase, 'RETURNING')

    def test_unsettled_route_never_claims_zero(self):
        e = self.e; now = time.monotonic()
        e.phase = 'RETURNING'; e.route_next = 'ZERO'; e.route_label = 'test'
        e.route = Route(np.zeros(14), [np.zeros(14)], e.limits, now - 20)
        e.state[0] = .1; e.settle_since = now - 10
        e._route_target(now)
        self.assertEqual(e.phase, 'ESTOP')

    def test_estop_resets_rate_filter_to_measured_hold(self):
        e = self.e; e.phase = 'RUNNING'; e.ep_start = time.monotonic(); e.state[0] = .2
        e._estop('test')
        np.testing.assert_allclose(e.safety.last, e.hold)

    def test_first_chunk_timeout_stops(self):
        e = self.e; e.warmup_pending = 0; e.phase = 'RUNNING'
        e.ep_start = e.plan_wait_since = time.monotonic() - 10
        e.scheduler.reset(e.ep_hold)
        e._target(time.monotonic())
        self.assertEqual(e.phase, 'ESTOP')

    def test_zero_during_running_is_honoured(self):
        e = self.e; e.phase = 'RUNNING'; e.ep_start = time.monotonic()
        e._handle('zero', None, time.monotonic())
        self.assertEqual(e.phase, 'RETURNING')

    def test_pause_resume_invalidates_inflight_generation(self):
        e = self.e; e.phase = 'RUNNING'; e.ep_start = time.monotonic()
        e.warmup_ok = True
        before = e.episode_id
        e._handle('pause', None, time.monotonic())
        e._resume(time.monotonic())
        self.assertGreater(e.episode_id, before)
        self.assertEqual(e.phase, 'RUNNING')

    def test_old_error_cannot_stop_new_generation(self):
        e = self.e; e.phase = 'RUNNING'; e.episode_id = 10
        e._handle('policy_error', (9, 'old failure'), time.monotonic())
        self.assertEqual(e.phase, 'RUNNING')

    def test_expired_plan_is_rejected(self):
        e = self.e; e.phase = 'RUNNING'; e.episode_id = 5; now = time.monotonic()
        points = np.zeros((50, 14))
        plan = Plan(now - 10, 30, 0, points, now)
        e._handle('policy_result', (5, plan, {'infer_ms': 10}, None), now)
        self.assertEqual(e.scheduler.target_at(now)[1]['n_plans'], 0)

    def test_stale_plan_cannot_enter_scheduler(self):
        e = self.e; e.phase = 'RUNNING'; e.episode_id = 5; now = time.monotonic()
        plan = Plan(now, 30, 0, np.zeros((50, 14)), now)
        e._handle('policy_result', (4, plan, {'infer_ms': 10}, None), now)
        self.assertEqual(e.scheduler.target_at(now)[1]['n_plans'], 0)

    def test_current_plan_enters_scheduler(self):
        e = self.e; e.phase = 'RUNNING'; e.episode_id = 5; now = time.monotonic()
        plan = Plan(now, 30, 0, np.zeros((50, 14)), now)
        e._handle('policy_result', (5, plan, {'infer_ms': 10}, None), now)
        self.assertEqual(e.scheduler.target_at(now)[1]['n_plans'], 1)

    def test_release_verifies_measured_pose(self):
        e = self.e; e.phase = 'RELEASING'; e.state[0] = .2
        e._do_release()
        self.assertNotEqual(e.phase, 'RELEASED')

    def test_fault_clears_old_release_request(self):
        e = self.e; e.release_requested = True
        e._enter_fault('lost feedback')
        self.assertFalse(e.release_requested)

    def test_late_first_plan_rejected_even_if_horizon_remains(self):
        e = self.e; e.phase = 'RUNNING'; e.episode_id = 5
        e.o.policy['first_plan_timeout_s'] = .1
        now = time.monotonic(); e.plan_wait_since = now - .2
        plan = Plan(now - .2, 30, 0, np.zeros((50, 14)), now)
        e._handle('policy_result', (5, plan, {}, None), now)
        self.assertEqual(e.phase, 'ESTOP')

    def test_exhausted_plan_holds_actual_instead_of_old_endpoint(self):
        e = self.e; e.phase = 'RUNNING'; e.ep_start = time.monotonic()
        e.plan_wait_since = e.ep_start; now = e.ep_start
        e.scheduler.reset(e.hold)
        e.scheduler.add(Plan(now - 49/30 - .01, 30, 0, np.full((50, 14), .1), now - .1))
        e.state[:] = 0
        target, _ = e._target(now)
        np.testing.assert_allclose(target, e.state)

    def test_camera_disconnect_invalidates_pending_generation(self):
        e = self.e; e.phase = 'RUNNING'; e.ep_start = time.monotonic()
        generation = e.episode_id
        e.plan_wait_since = e.ep_start
        e.scheduler.reset(e.hold)
        e.scheduler.add(Plan(e.ep_start, 30, 0, np.zeros((50,14)), e.ep_start))
        def grab(*args): raise RuntimeError('camera stale')
        e.cams = SimpleNamespace(grab=grab)
        e._tick()
        self.assertEqual(e.phase, 'ESTOP')
        self.assertGreater(e.episode_id, generation)


if __name__ == '__main__':
    unittest.main()
