"""Paired held-out, autoregressive Flow evaluation and unmodified HoloMotion replay."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest='stage', required=True)
    generate = commands.add_parser('generate')
    for name in ['data', 'text', 'run', 'out']:
        generate.add_argument('--' + name, required=True)
    track = commands.add_parser('track')
    track.add_argument('--out', required=True)
    track.add_argument('--onnx', required=True)
    render = commands.add_parser('render')
    render.add_argument('--out', required=True)
    render.add_argument('--render-cases', type=int, nargs='+', default=[0,3,6,9,12,15])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.stage == 'generate':
        import pyarrow.parquet as pq
        from omg.generation.bones_text import BonesTextCache
        from omg.generation.flow_reference import FlowReferenceGenerator
        clips = pq.read_table(Path(args.data) / 'clips.parquet').to_pylist()
        captions = json.loads((Path(args.data) / 'captions.json').read_text())
        rows = [sorted([r for r in clips if r['split'] == 'test' and r['frames'] >= 110
                        and word in captions[r['caption_ids'][0]].lower()],
                       key=lambda r: (-r['frames'], r['clip_id']))[0]
                for word in ['walk', 'run', 'dance', 'kick', 'jump', 'turn']]
        qpos = np.memmap(Path(args.data) / 'qpos.f32', mode='r', dtype=np.float32).reshape(-1, 36)
        cases = [dict(category=category, clip_id=r['clip_id'], caption=captions[r['caption_ids'][0]],
                      seed=seed, offset=r['offset'], frames=r['frames'])
                 for category, r in zip(['walk', 'run', 'dance', 'kick', 'jump', 'turn'], rows)
                 for seed in [1234, 2345, 3456]]
        history = torch.tensor(np.stack([qpos[c['offset']:c['offset']+10] for c in cases]), device='cuda')
        tokens, mask = BonesTextCache(args.text).lookup([c['caption'] for c in cases])
        conditions = dict(steps=10, text_tokens=tokens.cuda(), text_mask=mask.cuda())
        records = dict(cases=cases, fps=50, duration_s=20, replan_frames=25, committed_frames=25,
                       selection='Longest test clip per caption substring, tie-break clip_id; three fixed seeds',
                       history_source='Generated reference after the initial 10 ground-truth frames',
                       checkpoints={}, precision='bfloat16 autocast', ode_steps=10)
        np.save(out / 'history.npy', history.cpu().numpy())
        for i, c in enumerate(cases):
            np.save(out / f'gt_{i}.npy', qpos[c['offset']+10:c['offset']+min(c['frames'],1021)])
        for label in ['best', 'latest']:
            path = Path(args.run) / f'{label}.pt'
            digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
            saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            records['checkpoints'][label] = dict(step=saved['step'], sha256=digest,
                                                source_revision=saved.get('source_revision'),
                                                provenance=saved['provenance'])
            model = FlowReferenceGenerator.from_checkpoint(path, 'cuda')
            rngs = [torch.Generator(device='cuda').manual_seed(c['seed']) for c in cases]
            latencies = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                noise = torch.stack([torch.randn(100,38,generator=g,device='cuda') for g in rngs])
                plan = model.generate_suffix(history, history[:, :0], noise, **conditions)
                for cursor in range(25, 1000, 25):
                    noise = torch.stack([torch.randn(100,38,generator=g,device='cuda') for g in rngs])
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    suffix = model.generate_suffix(plan[:,cursor-10:cursor],plan[:,cursor:cursor+25],noise,**conditions)
                    torch.cuda.synchronize()
                    latencies.append(time.perf_counter()-start)
                    old = plan[:, :cursor+25].clone()
                    plan = torch.cat([old, suffix],dim=1)
                    assert torch.equal(plan[:, :cursor+25], old)
            raw = plan.float().cpu().numpy()
            assert np.isfinite(raw).all()
            np.save(out / f'{label}_reference.npy', raw)
            records['checkpoints'][label]['batch18_replan_ms_mean'] = float(np.mean(latencies)*1000)
            print(label, saved['step'], raw.shape, flush=True)
            del model, saved
        (out / 'manifest.json').write_text(json.dumps(records, indent=2))
    elif args.stage == 'track':
        import onnxruntime as ort
        from omg.tracking.holomotion.runner import HoloMotionRolloutRunner
        from omg.tracking.holomotion.runtime import g1_state_from_qpos_history
        record = json.loads((out / 'manifest.json').read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(args.onnx, sess_options=options, providers=['CPUExecutionProvider'])
        runner = HoloMotionRolloutRunner(holomotion_onnx=args.onnx, session=session, providers='CPUExecutionProvider')
        assert runner.metadata.n_fut_frames == 10
        assert abs(runner.model.opt.timestep * runner.control_substeps - .02) < 1e-9
        histories = np.load(out / 'history.npy')
        record['tracker'] = dict(onnx_sha256=hashlib.file_digest(open(args.onnx,'rb'),'sha256').hexdigest(),
                                onnx_path=args.onnx, action_clip=runner.action_clip,
                                control_substeps=runner.control_substeps, init='same observed pose and differentiated velocity',
                                failure_screen='first pelvis z < 0.35 m or root tilt > 60 degrees; no early termination',
                                limitation='Offline reference replay; does not test online latency or executed-state conditioning')
        rows = []
        for label in ['gt','best','latest']:
            reference = None if label == 'gt' else np.load(out / f'{label}_reference.npy')
            for i, case in enumerate(record['cases']):
                if label == 'gt' and case['seed'] != 1234:
                    continue
                ref = np.load(out / f'gt_{i}.npy') if label == 'gt' else reference[i]
                steps = min(1000, len(ref)-11)
                runner.clear_rollout_buffers(reset_initialized=True)
                pose, velocity = g1_state_from_qpos_history(runner.model,runner.g1_handles,histories[i],fps=50)
                executed = runner.run_reference_chunk(ref, reference_fps=50,steps=steps,
                                                       init_qpos_36=pose,init_qvel_35=velocity).qpos_36
                np.save(out / f'{label}_executed_{i}.npy', executed)
                np.save(out / f'{label}_actions_{i}.npy', np.asarray(runner.actions))
                tilt = 1 - 2*(executed[:,4]**2 + executed[:,5]**2)
                failed = np.flatnonzero((executed[:,2]<.35) | (tilt<.5))
                acc = np.abs(np.diff(ref[:steps,7:], n=2, axis=0))*2500
                seam_frames = np.arange(50, steps, 25)
                seam_idx = np.concatenate([seam_frames-2,seam_frames-1])
                result = dict(label=label,case=i,category=case['category'],seed=case['seed'],seconds=steps/50,
                              first_failure_s=None if not len(failed) else float(failed[0]/50),
                              pelvis_min_m=float(executed[:,2].min()),
                              joint_mae_rad=float(np.abs(executed[:,7:]-ref[:steps,7:]).mean()),
                              root_error_m=float(np.linalg.norm(executed[:,:3]-ref[:steps,:3],axis=1).mean()),
                              reference_joint_accel_mean=float(acc.mean()),
                              seam_joint_accel_mean=float(acc[seam_idx].mean()),
                              reference_root_height_min=float(ref[:steps,2].min()),
                              reference_root_height_max=float(ref[:steps,2].max()),
                              final_root_displacement_m=float(np.linalg.norm(ref[steps-1,:2]-ref[0,:2])))
                rows.append(result)
                (out / 'tracking.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(result),flush=True)
        runner.close()
        (out / 'manifest.json').write_text(json.dumps(record,indent=2))
    else:
        import cv2
        import imageio.v2 as imageio
        import mujoco
        from omg.tracking.holomotion.runtime import build_g1_state_handles, set_g1_qpos, resolve_robot_xml
        model = mujoco.MjModel.from_xml_path(str(resolve_robot_xml(None)))
        data = mujoco.MjData(model)
        handles = build_g1_state_handles(model)
        renderer = mujoco.Renderer(model,height=360,width=480)
        camera = mujoco.MjvCamera()
        camera.distance=4.5
        camera.azimuth=135
        camera.elevation=-20
        manifest=json.loads((out/'manifest.json').read_text())
        for i in args.render_cases:
            arrays=[np.load(out/'best_reference.npy')[i],np.load(out/'latest_reference.npy')[i],
                    np.load(out/f'best_executed_{i}.npy'),np.load(out/f'latest_executed_{i}.npy')]
            path=out/f"{manifest['cases'][i]['category']}_comparison.mp4"
            writer=imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=1)
            for frame in range(0,1000,2):
                panels=[]
                for panel, (arr,title) in enumerate(zip(arrays,['190.5k reference','200k reference','190.5k physics replay','200k physics replay'])):
                    set_g1_qpos(model,data,handles,arr[frame])
                    mujoco.mj_forward(model,data)
                    anchor=arrays[panel % 2][frame]
                    camera.lookat[:]=[anchor[0],anchor[1],.85]
                    renderer.update_scene(data,camera)
                    rgb=renderer.render().copy()
                    cv2.putText(rgb,title,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),2)
                    cv2.putText(rgb,f't={frame/50:.2f}s | seed 1234',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
                    if panel >= 2:
                        error=np.linalg.norm(arr[frame,:3]-anchor[:3])
                        cv2.putText(rgb,f'root error: {error:.2f} m',(10,75),cv2.FONT_HERSHEY_SIMPLEX,.5,(160,20,20),1)
                    panels.append(rgb)
                writer.append_data(np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0))
            writer.close()
            print(path,flush=True)
        renderer.close()


if __name__ == '__main__':
    main()
