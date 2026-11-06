"""Optional CPU torch/LeRobot processor acceptance; no network, tokenizer or real model.

python -m tests.optional_runner_smoke
"""
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
import torch
from pi_deploy.inference.inference import Pi05Runner
from pi_deploy.inference.options import gripper_options
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep, AbsoluteActionsProcessorStep
from lerobot.lerobot_types import TransitionKey
from tests.fixtures import checkpoint


def main():
    names = [f'right_joint{i}' for i in range(6)] + ['right_gripper']
    for velocity in (False, True):
        c = dict(action_names=names, use_velocity=velocity, prompt='test',
                 gripper_filter=dict(knee=.35, floor=.22, enabled=True))
        relative = RelativeActionsProcessorStep(enabled=True, exclude_joints=['gripper'], action_names=names)
        absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
        def pre(batch):
            t = relative({TransitionKey.OBSERVATION: {'observation.state': batch['observation.state']}})
            return t[TransitionKey.OBSERVATION]
        def post(actions):
            return absolute({TransitionKey.ACTION: actions})[TransitionKey.ACTION]
        policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=32))
        def predict(policy, batch, noise):
            out = torch.full((1, 50, 7), .1)
            out[..., 6] = -.1
            return out
        batch = {'observation.state': torch.full((1, 14), .5)}
        if velocity:
            batch['observation.velocity'] = torch.zeros((1, 14))
        for mode in ('on', 'off', 'checkpoint'):
            with patch('pi_deploy.inference.inference.load_checkpoint', return_value=(policy, pre, post, c)), \
                 patch('pi_deploy.inference.inference.predict_chunk', side_effect=predict):
                runner = Pi05Runner(Path('.'), Path('.'), 'cpu', gripper_options(c, mode))
                out = runner.predict(batch)
            np.testing.assert_allclose(out['filtered'][..., :6], .6, atol=1e-5)
            assert (out['filtered'][..., 6] == 0).all() if mode == 'off' else (out['filtered'][..., 6] > .22).all()
        print('Runner + real LeRobot relative/absolute processor + floor switches OK; velocity=',velocity)
    print('Processor acceptance only; neural network and actual checkpoint NOT tested')
    verify_strict_load_receipt()


def verify_strict_load_receipt():
    """Tiny local tensor verifies that swallowed upstream load errors are rejected."""
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from safetensors.torch import save_file, load_file
    from pi_deploy.inference.policy_factory import load_checkpoint

    class DummyPolicy(torch.nn.Module):
        fail = False
        def __init__(self, config):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.action_in_proj = torch.nn.Linear(1, 1, bias=False)
        @classmethod
        def from_pretrained(cls, path, config, **kwargs):
            instance = cls(config)
            try:
                sd = {'wrong_key': torch.ones(1)} if cls.fail else load_file(str(Path(path)/'model.safetensors'))
                instance.load_state_dict(sd, strict=True)
            except RuntimeError:
                pass  # reproduce upstream's exception swallowing
            return instance

    with tempfile.TemporaryDirectory() as temp:
        root = checkpoint(temp)
        save_file({'model.action_in_proj.weight': torch.tensor([[.75]])}, str(root/'model.safetensors'))
        config = PI05Config(device='cpu')
        with patch('lerobot.configs.PreTrainedConfig.from_pretrained', return_value=config), \
             patch('lerobot.policies.pi05.modeling_pi05.PI05Policy', DummyPolicy), \
             patch('pi_deploy.inference.policy_factory.build_processors', return_value=(None, None)):
            policy, _, _, _ = load_checkpoint(root, root, 'cpu')
            assert float(policy.model.action_in_proj.weight.detach()[0,0]) == .75
            DummyPolicy.fail = True
            try:
                load_checkpoint(root, root, 'cpu')
            except RuntimeError as exc:
                assert '严格加载' in str(exc)
            else:
                raise AssertionError('Swallowed load error must never return a policy')
    print('Strict load receipt + tensor probe OK (tiny local tensor, NOT actual Pi weights)')


if __name__ == '__main__':
    main()
