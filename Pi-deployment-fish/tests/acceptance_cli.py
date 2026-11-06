"""Acceptance of start.sh + model subprocess + scripted console with mock weights only.

Run with device Python: python -m tests.acceptance_cli [--bash path/to/bash]
Temporary fixture and configs are outside the real task weight folders.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import yaml
from tests.fixtures import checkpoint
from pi_deploy import PROJECT_ROOT
from pi_deploy.config import load_deploy, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bash')
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkpoint(root/'ckpt')
        dep = load_deploy()
        dep['policy_server']['port'] = 28890  # Windows may reserve the default Ubuntu port range.
        # Keep ACT's exact start pose; this acceptance actually traverses the full route.
        tasks = load_tasks(dep)
        task = tasks['blue_block']
        (root/'tasks.yaml').write_text(yaml.safe_dump({'tasks': {'cli_acceptance': dict(
            label='CLI acceptance MOCK', enabled=True, checkpoint_dirs=[str(root/'ckpt')],
            start_pose=task.start_pose, max_episode_s=80, execution='chunk_auto', model_overrides={})}}), encoding='utf-8')
        (root/'deploy.yaml').write_text(yaml.safe_dump(dep), encoding='utf-8')
        params = ['--deploy-config', str(root/'deploy.yaml'), '--tasks-config', str(root/'tasks.yaml'),
                  '--task', 'cli_acceptance', '--checkpoint', 'ckpt', '--mode', 'sim', '--preset', 'chunk_auto',
                  '--gripper-floor', 'off', '--backend', 'mock', '--device', 'cpu', '--yes', '--headless',
                  '--script', 'enter,wait:READY:40,enter,wait:RUNNING:10,sleep:0.7,p,wait:PAUSED:5,enter,wait:RUNNING:5,f,wait:ZERO:40,r']
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8', HF_HUB_OFFLINE='1',
                   DEVICE_PY=sys.executable, MODEL_PY=sys.executable)
        cmd = ([args.bash, str(PROJECT_ROOT/'start.sh')] if args.bash else [sys.executable, '-m', 'pi_deploy.app.main']) + params
        log = PROJECT_ROOT/'logs/cli_acceptance.log'
        with log.open('w', encoding='utf-8') as out:
            p = subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=100)
            except subprocess.TimeoutExpired:
                p.terminate()
                raise
        print(log.read_text(encoding='utf-8')[-4000:])
        if code:
            raise AssertionError(f'CLI exit {code}; log={log}')
        sessions = sorted((PROJECT_ROOT/'logs').glob('*_cli_acceptance_sim'), key=lambda p:p.stat().st_mtime)
        meta = json.loads((sessions[-1]/'session.json').read_text(encoding='utf-8'))
        assert meta['inference']['gripper_floor'] == 'off'
        assert meta['gripper_filter']['enabled'] is False
        events = (sessions[-1]/'events.log').read_text(encoding='utf-8')
        assert '电机已释放' in events and '实时预热完成' in events
        assert meta['backend'] == 'mock' and meta['simulated_hardware']
        print('CLI ACCEPTANCE OK (mock, no actual weights/GPU/hardware)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
