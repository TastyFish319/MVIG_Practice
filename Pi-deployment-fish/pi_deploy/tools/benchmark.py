"""Load an actual Pi checkpoint and benchmark GPU inference; NEVER opens robot hardware."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from pi_deploy.config import load_deploy, resolve_path
from pi_deploy.model.spec import resolve_spec
from pi_deploy.model.features import dummy_inputs
from pi_deploy.inference.options import gripper_options


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tokenizer')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--gripper-floor', choices=('checkpoint', 'on', 'off'), default='checkpoint')
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--observation', type=Path, help='可选 npz: state + 图像特征名；图像为已预处理 uint8 HWC')
    ap.add_argument('--output', type=Path, default=Path('logs/gpu_benchmark.json'))
    args = ap.parse_args()
    if args.iterations < 2 or args.warmup < 1:
        ap.error('iterations 至少2，warmup 至少1')
    cfg = load_deploy()
    spec = resolve_spec(args.checkpoint)
    tokenizer = args.tokenizer or str(resolve_path(cfg['inference']['tokenizer']))
    from pi_deploy.server.policy_server import LeRobotBackend, Server
    import torch
    start = time.perf_counter()
    backend = LeRobotBackend(spec.checkpoint, args.device, spec, tokenizer, args.gripper_floor,
                             cfg['inference']['seed'])
    load_s = time.perf_counter() - start
    state, images = dummy_inputs(spec)
    if args.observation:
        with np.load(args.observation, allow_pickle=False) as data:
            state = data['state'].copy()
            images = {im.key: data[im.key].copy() for im in spec.images}
    Server._check(type('Inputs', (), {'spec': spec})(), state, images)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    timings = []
    for i in range(args.warmup + args.iterations):
        t = time.perf_counter()
        out = backend.predict(state, images)
        ms = (time.perf_counter() - t) * 1000
        if out.shape != (spec.horizon, spec.action_dim) or not np.isfinite(out).all():
            raise ValueError('真实推理输出形状或数值非法')
        timings.append(ms)
        print(f'{i+1:02d}/{args.warmup+args.iterations} 推理 {ms:.1f}ms  shape={out.shape}', flush=True)
    stable = timings[args.warmup:]
    preset = cfg['execution']['presets'][cfg['execution']['default']]
    steps = preset.get('exec_steps') or spec.n_action_steps
    budget = ((spec.horizon - 1 - steps) / spec.fps - preset.get('blend_s', .1) - cfg['policy']['plan_margin_s'])
    p95 = float(np.percentile(stable, 95))
    report = dict(checkpoint=spec.checkpoint, backend=backend.info, load_s=load_s,
                  input_source=str(args.observation) if args.observation else 'synthetic (not task performance)',
                  first_ms=timings[0], warmup_ms=timings[:args.warmup], samples_ms=stable,
                  p50_ms=float(np.median(stable)), p95_ms=p95, output_shape=list(out.shape),
                  gripper_filter=gripper_options(spec.contract, args.gripper_floor),
                  peak_gpu_allocated_gb=(torch.cuda.max_memory_allocated(args.device)/1e9 if args.device.startswith('cuda') else None),
                  timing_budget_s=budget, timing_budget_ok=p95/1000 < budget,
                  note='仅模型耗时；本机TCP和真实相机/控制循环时延还需 shadow 验收',
                  fingerprint=spec.fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'报告: {args.output}；模型时延预算通过={report["timing_budget_ok"]}')
    return 0 if report['timing_budget_ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
