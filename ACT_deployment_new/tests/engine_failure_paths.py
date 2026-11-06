"""Engine failure paths with simulated hardware + mock model (~40 s, no torch needed).

    PYTHONPATH=. python tests/engine_failure_paths.py

inference failures -> ESTOP -> emergency zero; episode time limit -> zero; worker FAULT -> recover."""
import time, tempfile, numpy as np
from pathlib import Path
from act_deploy.config import load_deploy, load_tasks
from act_deploy.model.spec import resolve_spec
from act_deploy.app.engine import Engine, EngineOptions
from act_deploy.hw.arms import SimArms, ArmFault
from act_deploy.hw.cameras import CameraRig
from act_deploy.server.policy_server import MockBackend

dep = load_deploy(); task = load_tasks(dep)["block_into_cup"]
spec = resolve_spec("checkpoints/block_into_cup/v3_long2_017000")
dep["motion"].update(max_joint_velocity=1.0, max_joint_acceleration=4.0, min_duration_s=0.5)

class Client:
    def __init__(s): s.n=0; s.fail_after=None; s.b=MockBackend(spec, 0.01)
    def predict(s, st, im):
        s.n+=1
        if s.fail_after is not None and s.n > s.fail_after: raise TimeoutError("simulated timeout")
        return s.b.predict(st, im), {"infer_ms": 10}

class FaultyArms(SimArms):
    fault_once=False
    def read(s, allow_fault=False):
        x=super().read()
        if s.fault_once:
            s.fault=" right: command lease expired"
            if not allow_fault: raise ArmFault(s.fault)
        return x
    def reset_fault(s): s.fault_once=False; s.fault=None; super().reset_fault()

def mk(arms, client):
    d=Path(tempfile.mkdtemp())
    o=EngineOptions(mode="real", rate_hz=100, motion=dep["motion"], poses=dep["poses"], safety=dep["safety"],
        policy=dep["policy"], preset=dep["execution"]["presets"]["chunk_10"], recording=dep["recording"],
        max_frame_age_s=0.2, start_pose=np.asarray(task.start_pose), max_episode_s=4.0, session_dir=d, meta={})
    cams=CameraRig.fake(["left_wrist","right_wrist"],640,480,30); cams.wait_ready()
    e=Engine(o, spec, arms, cams, client); e.start(); return e
def wait(e, ph, t=30):
    end=time.time()+t
    while e.phase!=ph:
        if time.time()>end: raise AssertionError(f"wanted {ph}, at {e.phase}: {list(e.events)[-3:]}")
        time.sleep(0.02)

c=Client(); e=mk(SimArms(np.asarray(dep["poses"]["zero"])), c)
e.command("next"); wait(e,"READY"); print("0 warmup:", e.warmup_report)
c.fail_after=c.n+3; e.command("next"); wait(e,"ESTOP",10)
print("1 ESTOP after failures:", [m for _,l,m in e.events if l=="error"][-1])
e.command("zero"); wait(e,"ZERO"); print("1 emergency zero ok, R state", np.round(e.state[7:],2))
c.fail_after=None; e.command("next"); wait(e,"READY"); e.command("next"); wait(e,"RUNNING"); wait(e,"RETURNING",8); wait(e,"ZERO")
print("2 time limit:", e.results[-1]["reason"])
e.command("release"); wait(e,"RELEASED"); e.cams.close()
a=FaultyArms(np.asarray(dep["poses"]["zero"])); e=mk(a, Client())
e.command("next"); wait(e,"READY"); e.command("next"); wait(e,"RUNNING"); time.sleep(1); a.fault_once=True; wait(e,"FAULT",3)
print("3 FAULT:", e.fault_reason)
e.command("zero"); wait(e,"ZERO"); print("3 recovered ->", e.phase, "| episode:", e.results[-1]["reason"])
e.command("release"); wait(e,"RELEASED"); e.cams.close(); print("ALL OK")
