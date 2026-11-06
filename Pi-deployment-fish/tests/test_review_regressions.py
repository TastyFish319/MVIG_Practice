"""Focused regression checks for the final independent review; no physical hardware."""
import contextlib
import io
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np

from pi_deploy.app import main as app
from pi_deploy.tools import rescue
from pi_deploy.config import load_deploy, load_tasks
from pi_deploy.hw.arms import FollowerArms
from pi_deploy.model.spec import resolve_spec
from tests.fixtures import checkpoint


class ReviewRegressions(unittest.TestCase):
    def test_wrong_model_process_is_not_accepted_or_shutdown(self):
        srv = app.ServerProcess.__new__(app.ServerProcess)
        srv.proc = Mock(pid=123); srv.proc.poll.return_value = None
        srv.client = Mock(); srv.log = Mock(); srv.verified = False
        srv.instance_id = 'own'; srv.expected_backend = 'mock'; srv.expected_gripper = {}
        srv.client.health.return_value = {'pid': 456, 'instance_id': 'other'}
        with self.assertRaises(RuntimeError):
            srv.health()
        srv.stop()
        srv.client.shutdown.assert_not_called()

    def test_force_release_cli_is_removed(self):
        with patch('sys.argv', ['rescue', '--release-now']), contextlib.redirect_stderr(io.StringIO()), \
             patch.object(rescue, 'FollowerArms') as arm:
            with self.assertRaises(SystemExit) as ex:
                rescue.main()
            self.assertEqual(ex.exception.code, 2)
            arm.assert_not_called()

    def test_release_uses_verified_pidfd_not_replaced_pid_file(self):
        arm = FollowerArms.__new__(FollowerArms)
        identity = {'pid': 123, 'instance': 'own'}
        arm.identities = {'right': identity}; arm.pid_handles = {'right': 456}
        arm.procs = {}; arm.clients = {}; arm.end = Mock()
        arm._call = Mock(return_value=identity)
        with patch('pi_deploy.hw.arms.signal.pidfd_send_signal', create=True) as send, \
             patch('pi_deploy.hw.arms.os.close') as close, patch('pi_deploy.hw.arms.pid_alive', return_value=False):
            arm.release()
            send.assert_called_once_with(456, signal.SIGTERM)
            close.assert_called_once_with(456)

    def test_single_arm_rescue_reads_after_confirmation_and_checks_zero(self):
        dep = load_deploy(); dep['motion'].update(min_duration_s=.01, max_joint_velocity=10, max_joint_acceleration=100)
        state = np.zeros(14); state[13] = .99; state[7] = .01
        begin = []
        arm = SimpleNamespace(sides=('right',), fault=None,
                              read=lambda **kw: state.copy(), begin=lambda x: begin.append(x.copy()),
                              submit=lambda x: np.copyto(state, x), release=Mock(), end=Mock())
        def confirm(*args):
            state[7] = .02  # Pose changed while the operator considered the prompt.
            return 'y'
        with patch('sys.argv', ['rescue']), patch.object(rescue, 'leftover_workers', return_value=[{'side':'right'}]), \
             patch.object(rescue, 'load_ports', return_value={'right':1235}), patch.object(rescue, 'port_open', return_value=True), \
             patch.object(rescue, 'load_deploy', return_value=dep), patch.object(rescue, 'FollowerArms', return_value=arm) as ctor, \
             patch('builtins.input', side_effect=confirm), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rescue.main(), 0)
        self.assertEqual(ctor.call_args.kwargs['sides'], ('right',))
        self.assertEqual(begin[0][7], .02)
        arm.release.assert_called_once()
        np.testing.assert_allclose(state[7:13], 0)

    def test_console_exception_detaches_without_requesting_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = resolve_spec(checkpoint(root/'ckpt'))
            dep = load_deploy(); task = load_tasks(dep)['blue_block']
            ckpt = SimpleNamespace(name='ckpt', path=root/'ckpt')
            engine = Mock(phase='RUNNING', _save_threads=[])
            engine.done.is_set.return_value = False
            camera = Mock(); server = Mock()
            with patch('sys.argv', ['main', '--mode', 'sim', '--backend', 'mock', '--yes']), \
                 patch.object(app, 'load_deploy', return_value=dep), patch.object(app, 'load_tasks', return_value={task.name: task}), \
                 patch.object(app, 'pick', return_value=(task, ckpt, spec, 'sim', 'chunk_auto', 'off')), \
                 patch.object(app, 'LOG_ROOT', root/'logs'), patch.object(app, 'ServerProcess', return_value=server), \
                 patch.object(app, 'wait_server', return_value={}), \
                 patch('pi_deploy.hw.cameras.CameraRig.fake', return_value=camera), \
                 patch('pi_deploy.app.engine.Engine', return_value=engine), \
                 patch.object(app, 'run_console', side_effect=RuntimeError('dashboard failed')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(app.main(), 1)
            commands = [c.args[0] for c in engine.command.call_args_list]
            self.assertIn('detach', commands)
            self.assertNotIn('release', commands)


if __name__ == '__main__':
    unittest.main()
