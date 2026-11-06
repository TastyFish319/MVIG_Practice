"""Small metadata-only fixtures; never pretend these are trained weights."""
import json
import struct
from pathlib import Path


def checkpoint(root, *, both=False, velocity=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    names = [f'{arm}_joint{i}' for arm in ('right', 'left') for i in range(6)]
    names = [f'{a}_{n}' for a in ('right', 'left') for n in [*[f'joint{i}' for i in range(6)], 'gripper']]
    action = names if both else names[:7]
    contract = dict(fps=30, use_velocity=velocity, state_names=names + ([n + '_vel' for n in names] if velocity else []),
                    action_names=action, chunk_size=50, n_action_steps=15,
                    image_keys=['observation.images.right_wrist', 'observation.images.left_wrist'],
                    prompt='Pick the blue block.', gripper_filter=dict(knee=0.35, floor=0.22, enabled=True),
                    action_representation='关节: 相对(leader - 当前follower)；夹爪: 绝对 leader')
    inputs = {f'observation.images.{s}_wrist': dict(type='VISUAL', shape=[3, 224, 224]) for s in ('left', 'right')}
    inputs['observation.state'] = dict(type='STATE', shape=[28 if velocity else 14])
    cfg = dict(type='pi05', input_features=inputs, output_features={'action': dict(type='ACTION', shape=[len(action)])},
               chunk_size=50, n_action_steps=15, use_relative_actions=True, relative_exclude_joints=['gripper'])
    for name, data in [('config.json', cfg), ('pitrain_contract.json', contract),
                       ('policy_preprocessor.json', {'steps': [{'registry_name': 'normalizer_processor', 'state_file': 'pre.safetensors'}]}),
                       ('policy_postprocessor.json', {'steps': [{'registry_name': 'unnormalizer_processor', 'state_file': 'post.safetensors'}]})]:
        (root / name).write_text(json.dumps(data), encoding='utf-8')
    for name in ('model.safetensors', 'pre.safetensors', 'post.safetensors'):
        (root / name).write_bytes(struct.pack('<Q', 2) + b'{}')
    return root

