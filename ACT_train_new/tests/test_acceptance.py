"""最终集中验收：手算数值、边界、失败发布和独立入口。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from acttrain.data.core import (causal_indices, split_episodes, load_selection, file_hash,
                               write_json, numeric_statistics)
from acttrain.model.relative import relative_actions, absolute_actions, relative_statistics, launch_coordinates
from acttrain.model.velocity import augment_states
from acttrain.model.gripper import soft_floor
from acttrain.data.records import mcap_paths
from acttrain.training.config import TrainingConfig
from acttrain.training.pipeline import training_command, evaluation_command, retain_checkpoints
from acttrain.evaluation.metrics import error_values, report_from_errors, ranking_score, ranking_components
from acttrain.evaluation.parallel import evaluation_devices, checkpoint_assignments, run_parallel_evaluation

class Contracts(unittest.TestCase):
    def test_one_anchor_and_absolute_gripper(self):
        state=np.zeros(14);state[7:13]=1
        action=np.ones((3,7));action[:,:6]=[[1.2]*6,[1.3]*6,[1.1]*6];action[:,6]=[0,.5,1]
        delta=relative_actions(action,state)
        np.testing.assert_allclose(delta[:,:6],np.array([[.2]*6,[.3]*6,[.1]*6]))
        np.testing.assert_array_equal(delta[:,6],action[:,6])
        np.testing.assert_allclose(absolute_actions(delta,state),action)

    def test_velocity_only_past_and_episode_start(self):
        state=np.zeros((5,14));state[:,7:14]=np.arange(5)[:,None]*.1
        result=augment_states(state,np.arange(5,dtype=np.int64)*50_000_000)
        np.testing.assert_array_equal(result[:,:14],state.astype('float32'))
        np.testing.assert_array_equal(result[:2,14:],0)
        np.testing.assert_allclose(result[2:,14:],2,atol=1e-6)
        np.testing.assert_array_equal(augment_states(state[3:])[:,14:],0)
        with self.assertRaises(ValueError):augment_states(state,np.arange(5)*51_000_000)

    def test_statistics_train_legal_windows_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=np.zeros((33,14));action=np.zeros((33,7));action[:,0]=np.arange(33)
            np.savez(root/'train.npz',state=state,action=action)
            np.savez(root/'val.npz',state=state+1000,action=action+1000)
            stats=relative_statistics(root,[{'split':'train','trace':'train.npz','rows':33},
                                          {'split':'val','trace':'val.npz','rows':33}])
            self.assertEqual(stats['count'],[64]);self.assertAlmostEqual(stats['mean'][0],16)
            self.assertGreaterEqual(min(stats['std']),1e-3)

    def test_causal_alignment_rejects_future_and_stale(self):
        np.testing.assert_array_equal(causal_indices([0,10,20],[5,10,21],10),[0,1,2])
        for times,targets in (([10,20],[0]),([0,10],[30]),([0,0],[0])):
            with self.assertRaises(ValueError):causal_indices(times,targets,10)

    def test_stratified_split_and_manual_membership(self):
        names=[f'task/episode_{i:04d}' for i in range(50)]
        split=split_episodes(names,42,.2);self.assertEqual(split,split_episodes(names,42,.2))
        chosen=[i for i,n in enumerate(names) if split[n]=='val']
        self.assertEqual(len(chosen),10)
        self.assertEqual([i//5 for i in chosen],list(range(10)))
        with self.assertRaises(ValueError):split_episodes(names,42,.2,['outside'])

    def test_review_inventory_hash_and_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'task/episode_0001/data';folder.mkdir(parents=True)
            (folder/'a.mcap').write_bytes(b'reviewed')
            record={'source_relative':'task/episode_0001','decision':'approved','technical_status':'pass',
                    'source_files':[{'name':'a.mcap','sha256':file_hash(folder/'a.mcap')}]}
            approved={'task':'task','source_fingerprint':'fingerprint','approved_episodes':[record['source_relative']]}
            review={'task':'task','source_fingerprint':'fingerprint','episodes':[record]}
            write_json(root/'approved.json',approved);write_json(root/'review.json',review)
            self.assertEqual(load_selection(root,root/'approved.json',root/'review.json')[1][0]['relative'],record['source_relative'])
            (folder/'a.mcap').write_bytes(b'changed')
            with self.assertRaises(ValueError):load_selection(root,root/'approved.json',root/'review.json')
            record['source_relative']='../outside/episode_0001';approved['approved_episodes']=[record['source_relative']]
            write_json(root/'approved.json',approved);write_json(root/'review.json',review)
            with self.assertRaises(ValueError):load_selection(root,root/'approved.json',root/'review.json')

    def test_mcap_inventory_rejects_escape_and_duplicate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir();(root/'data/a.mcap').touch()
            (root/'data/b.mcap').touch()
            for names in (['../escape.mcap'],['a.mcap','a.mcap'],['a.mcap']):
                write_json(root/'data/mcap_index.json',[{'filename':n} for n in names])
                with self.assertRaises(ValueError):mcap_paths(root)

    @unittest.skipUnless(sys.platform=='linux','目录软链接边界在Linux验收')
    def test_data_directory_symlink_cannot_escape_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode=root/'episode';outside=root/'outside'
            episode.mkdir();outside.mkdir();(outside/'a.mcap').touch()
            (episode/'data').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError):mcap_paths(episode)

    def test_soft_floor_continuous_monotone(self):
        values=np.linspace(-1,1,1000);filtered=soft_floor(values,.15,.3)
        self.assertTrue(np.all(np.diff(filtered)>0));self.assertTrue(np.all(filtered>.15))
        self.assertAlmostEqual(float(soft_floor([.3],.15,.3)[0]),.3)
        np.testing.assert_array_equal(soft_floor([.5,1],.15,.3),[.5,1])
        with self.assertRaises(ValueError):soft_floor([0],.4,.3)

    def test_tail_mask_and_compatible_ranking(self):
        target=np.zeros((2,7));chunks=np.ones((2,32,7));chunks[:,2:]=1e6
        error=error_values(chunks,target)
        self.assertEqual(error['full'].shape,(3,7));np.testing.assert_array_equal(error['full'],1)
        report=report_from_errors(error,np.ones(7))
        self.assertEqual(ranking_score(ranking_components(report,np.ones(7))),1)
        self.assertEqual(report['full']['axes']['right_joint0']['p95_abs'],1)

    def test_launch_counts_and_devices(self):
        for n in (1,2,4):
            self.assertEqual(launch_coordinates({'WORLD_SIZE':str(n),'LOCAL_WORLD_SIZE':str(n),'RANK':'0','LOCAL_RANK':'0'},n),(0,0))
            self.assertEqual(len(evaluation_devices('cuda',n,7,4)),n)
        self.assertEqual(evaluation_devices('cuda',4,2,4),['cuda:0','cuda:1'])
        with self.assertRaises(ValueError):evaluation_devices('cuda',4,7,2)
        self.assertEqual(sorted(sum(checkpoint_assignments(list(range(7)),['cuda:0','cuda:1']),[])),list(range(7)))

class Engineering(unittest.TestCase):
    def test_strict_configuration(self):
        for bad in ({'unknown':1},{'steps':True},{'steps':0},{'gpus':3},{'lr':float('nan')},{'use_velocity':'false'},{'eval_max_episodes':-1}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):TrainingConfig.from_dict(bad)
        self.assertEqual(TrainingConfig.from_dict({}).dropout,.05)

    def test_train_eval_plan_same_devices_and_velocity(self):
        config=TrainingConfig(gpus=4,use_velocity=False,policy_steplr=True)
        train=training_command(config,Path('/data'),Path('/run'));evaluation=evaluation_command(config,Path('/data'),Path('/run'))
        self.assertIn('--nproc-per-node=4',train);self.assertIn('--no-use-velocity',train)
        self.assertEqual(evaluation[evaluation.index('--gpus')+1],'4')
        self.assertIn('--policy-steplr',train);self.assertIn('--no-backbone-steplr',train)

    def test_group_schedule_boundaries(self):
        import torch
        from acttrain.training.schedule import GroupStepLRSchedulerConfig
        params=[torch.nn.Parameter(torch.zeros(1)),torch.nn.Parameter(torch.zeros(1))]
        optimizer=torch.optim.AdamW([{'params':[params[0]],'lr':.01},{'params':[params[1]],'lr':.001}])
        scheduler=GroupStepLRSchedulerConfig(policy_enabled=True,policy_step_size=2,policy_gamma=.5).build(optimizer,4)
        for step in range(4):
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],.01*.5**(step//2))
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'],.001)
            optimizer.step();scheduler.step()

    def test_checkpoint_retention_keeps_best_and_final(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'metrics').mkdir()
            (root/'metrics/validation_metrics.csv').write_text('step,val_loss\n1,0.1\n2,0.3\n3,0.2\n')
            for step in range(1,4):(root/'checkpoints'/f'{step:06}').mkdir(parents=True)
            retain_checkpoints(root,1)
            self.assertTrue((root/'checkpoints/000001').is_dir());self.assertTrue((root/'checkpoints/000003').is_dir())
            self.assertFalse((root/'checkpoints/000002').exists())

    def test_parallel_failure_does_not_publish_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);script=root/'fail.py';script.write_text('raise RuntimeError("intentional failure")')
            with patch('acttrain.evaluation.parallel.worker_command',return_value=[sys.executable,str(script)]):
                with self.assertRaises(RuntimeError):run_parallel_evaluation({'output':str(root)},[{'index':0},{'index':1}],['cuda:0','cuda:1'])
            self.assertEqual(json.loads((root/'progress.json').read_text())['status'],'failed')
            self.assertFalse((root/'best_checkpoint.json').exists())

    @unittest.skipUnless(sys.platform=='linux','进程组停止验收只适用Linux')
    def test_signal_during_spawn_cleans_child_and_restores_handlers(self):
        import signal
        from acttrain.training.pipeline import run
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);plan=root/'plan.json'
            write_json(plan,{'config':{},'output':str(root/'run'),'dataset':str(root/'data'),'manifest_sha256':'test'})
            original=subprocess.Popen;children=[];handler=signal.getsignal(signal.SIGTERM)
            def spawn(*args,**kwargs):
                process=original(*args,**kwargs);children.append(process)
                signal.raise_signal(signal.SIGTERM)
                return process
            with patch('acttrain.training.pipeline.training_command',return_value=[sys.executable,'-c','import time;time.sleep(60)']),patch('acttrain.training.pipeline.subprocess.Popen',side_effect=spawn):
                with self.assertRaises(InterruptedError):run(plan)
            self.assertEqual(len(children),1);self.assertIsNotNone(children[0].poll())
            self.assertEqual(signal.getsignal(signal.SIGTERM),handler)
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'failed')

    def test_existing_partial_status_is_never_modified(self):
        from acttrain.data.convert import convert
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/preprocess.example.json').read_text(encoding='utf-8'))
            cfg['output']=str(root/'dataset')
            write_json(root/'config.json',cfg)
            stage=root/'dataset.partial';stage.mkdir()
            original=b'{"status":"previous_failure","owner":"another_attempt"}'
            (stage/'status.json').write_bytes(original)
            with self.assertRaises(FileExistsError):convert(root,root/'config.json')
            self.assertEqual((stage/'status.json').read_bytes(),original)

    def test_existing_output_and_invalid_config_do_not_start_job(self):
        from acttrain.cli.common import safe_name
        for name in ('../bad','/abs','bad name','..'):
            with self.assertRaises(ValueError):safe_name(name)
        from acttrain.cli.preprocess import validate_config
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/preprocess.example.json').read_text(encoding='utf-8'))
        for bad in ({'fps':30},{'action_source':'follower'},{'val_fraction':float('nan')}):
            with self.assertRaises(ValueError):validate_config({**cfg,**bad})

if __name__=='__main__':unittest.main()
