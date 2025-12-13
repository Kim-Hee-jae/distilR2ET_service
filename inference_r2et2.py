# -*- coding: cp949 -*-

import os
import sys
sys.path.append("./outside-code")

import yaml
import numpy as np
import torch
import torch.nn as nn

import argparse
from tqdm import tqdm

from os import listdir, makedirs
from os.path import exists, join, splitext
from pathlib import Path
from src.model_shape_aware import RetNet
from inference_bvh import getmodel
import time 


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='R2ET for motion retargeting')
    parser.add_argument(
        '--config',
        default='./config/inference_cfg.yaml',
        help='path to the configuration file',
    )
    parser.add_argument(
        '--save_path', default='./', help='path to the configuration file'
    )
    parser.add_argument('--phase', default='test', help='train or test')
    parser.add_argument('--load_inp_data', type=dict, default=dict(), help='')
    parser.add_argument('--weights', default='', help='xxx.pt weights for generator')
    parser.add_argument(
        '--device', type=int, default=0, nargs='+', help='only 0 avaliable'
    )
    parser.add_argument(
        '--num_joint', type=int, default=22, help='number of the joints'
    )
    parser.add_argument(
        '--ret_model_args',
        type=dict,
        default=dict(),
        help='the arguments of retargetor',
    )
    parser.add_argument(
        '--k', type=float, default=0.8, help='adjustable k for balacing gate'
    )

    parser.add_argument(
        '--job_id',
        default='',
        help='service/jobs/<job_id>',
    )

    return parser


def _get_height_from_skel(skel_frame: np.ndarray) -> float:
    diffs = np.sqrt((skel_frame ** 2).sum(axis=-1))
    height = diffs[1:6].sum() + diffs[7:10].sum()
    return float(height)


def load_from_preprocessed_mixamo(
    device,
    src_char: str,
    motion_name: str,
    num_joint: int,
    inp_shape_dir_path: str,
    tgt_shape_path: str,
    stats_path: str,
    inp_q_dir_path: str,
) :
    """
    output:
        inp_seq, inpskel, tgtskel,
        inp_shape, tgt_shape,
        inpquat,
        inp_height, tgt_height,
        local_mean, local_std,
        quat_mean, quat_std,
        global_mean, global_std,
        tgtanim, tgtname, tgtftime,
        inpanim, inpname, inpftime,
        inpjoints, tgtjoints
    """

    # 1) 통계 로드
    local_mean = np.load(join(stats_path, "mixamo_local_motion_mean.npy")).astype(np.float32)
    local_std  = np.load(join(stats_path, "mixamo_local_motion_std.npy")).astype(np.float32)
    global_mean = np.load(join(stats_path, "mixamo_global_motion_mean.npy")).astype(np.float32)
    global_std  = np.load(join(stats_path, "mixamo_global_motion_std.npy")).astype(np.float32)
    quat_mean  = np.load(join(stats_path, "mixamo_quat_mean.npy")).astype(np.float32)
    quat_std   = np.load(join(stats_path, "mixamo_quat_std.npy")).astype(np.float32)

    # 2) shape 로드
    def _load_shape(shape_npz_path: str) -> np.ndarray:
        f = np.load(shape_npz_path)
        full_width = f["full_width"].astype(np.single)
        joint_shape = f["joint_shape"].astype(np.single)
        shape_vec = np.divide(joint_shape, full_width[None, :])
        return shape_vec 

    inp_shape_path = join(inp_shape_dir_path, src_char + ".npz")
    inp_shape_vec = _load_shape(inp_shape_path)
    tgt_shape_vec = _load_shape(tgt_shape_path)

    # (1, shape_dim) 로 flatten
    inp_shape = torch.from_numpy(inp_shape_vec.reshape(1, -1)).float().to(device)
    tgt_shape = torch.from_numpy(tgt_shape_vec.reshape(1, -1)).float().to(device)

    #  3) motion SEQ / SKEL / QUAT 로드 
    motion_root = join(inp_q_dir_path, src_char)
    base = motion_name  # 확장자 없는 이름

    seq_path   = join(motion_root, base + "_seq.npy")
    skel_path  = join(motion_root, base + "_skel.npy")
    quat_path  = join(motion_root, base + "_quat.npy")

    if not (os.path.exists(seq_path) and os.path.exists(skel_path) and os.path.exists(quat_path)):
        raise FileNotFoundError(f"Preprocessed files not found for {src_char}/{base}: "
                                f"{seq_path}, {skel_path}, {quat_path}")

    seq  = np.load(seq_path)   # (T, J*3+8)
    skel = np.load(skel_path)  # (T, J, 3)
    quat = np.load(quat_path)  # (T, J, 4)

    T = seq.shape[0]

    # 4) local / global 분리
    local = seq[:, :-8].reshape(T, num_joint, 3)   # (T, J, 3)
    global_off = seq[:, -8:-4]                     # (T, 4)

    skel[:, 0, :] = local[:, 0, :]

    # 5) 정규화 (train 과 동일)
    # local_mean, local_std: (1, J, 3)
    # quat_mean, quat_std  : (1, J, 4)
    local_std[local_std == 0] = 1
    quat_std[quat_std == 0] = 1

    local_norm = (local - local_mean) / local_std
    skel_norm  = (skel  - local_mean) / local_std
    quat_norm  = (quat  - quat_mean) / quat_std

    # 6) Height 계산
    joints_a_world = skel_norm[0:1] * local_std + local_mean  # (1, J, 3)
    height_a = _get_height_from_skel(joints_a_world[0]) / 100.0  # cm -> m

    # target 캐릭터 이름
    tgt_char = Path(tgt_shape_path).stem
    tgt_skel_dir = join(inp_q_dir_path, tgt_char)
    tgt_skel_path = join(tgt_skel_dir, base + "_skel.npy")

    if os.path.exists(tgt_skel_path):
        skelB = np.load(tgt_skel_path).astype(np.float32)  # (T, J, 3)
        # frame 0 기준 키
        joints_b_world = skelB[0:1]  #  world 좌표 가정
    else:
        # source height 기반 대략적인 비율로만 맞추고
        # 스켈레톤은 일단 source 와 동일하게 사용
        joints_b_world = joints_a_world

    height_b = _get_height_from_skel(joints_b_world[0]) / 100.0

    inp_height = torch.tensor([[height_a]], dtype=torch.float32, device=device)
    tgt_height = torch.tensor([[height_b]], dtype=torch.float32, device=device)

    # 7) R2ET 입력 텐서 구성 
    # inp_seq : (1, T, J*3 + 4)
    local_flat  = local_norm.reshape(T, -1).astype(np.float32)   # (T, J*3)
    global_flat = global_off.astype(np.float32)                  # (T, 4)
    inp_seq_np  = np.concatenate([local_flat, global_flat], axis=-1)  # (T, J*3+4)

    # inpskel : (1, T, J*3)
    inpskel_np = skel_norm.reshape(T, -1).astype(np.float32)

    inp_seq  = torch.from_numpy(inp_seq_np[None]).float().to(device)   # (1, T, J*3+4)
    inpskel  = torch.from_numpy(inpskel_np[None]).float().to(device)   # (1, T, J*3)
    inpquat  = torch.from_numpy(quat_norm[None]).float().to(device)    # (1, T, J, 4)

    #  8) target skeleton (tgtskel)
    #  - Claire 의 스켈레톤을 동일한 방식으로 정규화,
    #    한 프레임(T-pose)만 있으면 길이 T 로 repeat 해서 사용.
    if os.path.exists(tgt_skel_path):
        tgt_skel = np.load(tgt_skel_path).astype(np.float32)  # (T' or 1, J, 3)
    else:
        tgt_skel = joints_b_world  # (1, J, 3)

    tgt_skel_norm = (tgt_skel - local_mean) / local_std  # (T' or 1, J, 3)

    if tgt_skel_norm.shape[0] == 1 and T > 1:
        tgt_skel_norm = np.repeat(tgt_skel_norm, T, axis=0)  # (T, J, 3)
    elif tgt_skel_norm.shape[0] != T:
        # 길이가 다르면 최소 길이에 맞춰 자름
        min_T = min(T, tgt_skel_norm.shape[0])
        tgt_skel_norm = tgt_skel_norm[:min_T]
        inp_seq  = inp_seq[:, :min_T]
        inpskel  = inpskel[:, :min_T]
        inpquat  = inpquat[:, :min_T]

    tgtskel_np = tgt_skel_norm.reshape(tgt_skel_norm.shape[0], -1).astype(np.float32)
    tgtskel = torch.from_numpy(tgtskel_np[None]).float().to(device)    # (1, T, J*3)

    # 9) 나머지 반환값 (호환용 dummy)
    local_mean_t  = local_mean # torch.from_numpy(local_mean).float().to(device)
    local_std_t   = local_std # torch.from_numpy(local_std).float().to(device)
    quat_mean_t   = quat_mean # torch.from_numpy(quat_mean).float().to(device)
    quat_std_t    = quat_std # torch.from_numpy(quat_std).float().to(device)
    global_mean_t = global_mean # torch.from_numpy(global_mean).float().to(device)
    global_std_t  = global_std # torch.from_numpy(global_std).float().to(device)

    # inference_r2et.py 의 load_from_bvh 와 인터페이스만 맞추기 위한 dummy 값들
    tgtanim = None
    tgtname = tgt_char
    tgtftime = None
    inpanim = None
    inpname = motion_name
    inpftime = None
    inpjoints = None
    tgtjoints = None

    return (
        inp_seq,
        inpskel,
        tgtskel,
        inp_shape,
        tgt_shape,
        inpquat,
        inp_height,
        tgt_height,
        local_mean_t,
        local_std_t,
        quat_mean_t,
        quat_std_t,
        global_mean_t,
        global_std_t,
        tgtanim,
        tgtname,
        tgtftime,
        inpanim,
        inpname,
        inpftime,
        inpjoints,
        tgtjoints,
    )

def main(arg):
    ret_model = getmodel(arg.weights, arg)
    parents = np.array(
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20]
    )

    inp_bvh_root = arg.load_inp_data['inp_q_dir_path']
    src_char_list = listdir(inp_bvh_root)

    # calc steps
    total_steps = 0
    for src_char in src_char_list:
        save_dir_path = join(arg.save_path, src_char)
        if exists(save_dir_path):
            continue

        bvh_dir_path = join(inp_bvh_root, src_char)
        if not exists(bvh_dir_path):
            continue

        bvh_list = [
            bvh_name for bvh_name in listdir(bvh_dir_path)
            if bvh_name.endswith('_seq.npy')
        ]
        total_steps += len(bvh_list)

    print(f"R2ET_TOTAL_STEPS {total_steps}", flush=True)

    # inference
    processed_steps = 0
    start_time = time.time()

    for src_char in src_char_list:
        print(f'Source Character: {src_char}')
        save_dir_path = join(arg.save_path, src_char)
        if exists(save_dir_path):
            print(f'Folder: {save_dir_path} already exists.')
            print(f'Skip {src_char}')
            continue
        else:
            makedirs(save_dir_path)

        seq_dir_path = join(arg.load_inp_data['inp_q_dir_path'], src_char)
        seq_list = [
            seq_name[:-8] for seq_name in listdir(seq_dir_path)
            if seq_name.endswith('_seq.npy')
        ] # 확장자 없음

        for seq_name in seq_list:
            save_path = join(save_dir_path, f'{seq_name}.npz')
            motion_name = seq_name

            # load data
            (
            inp_seq,
            inpskel,
            tgtskel,
            inp_shape,
            tgt_shape,
            inpquat,
            inp_height,
            tgt_height,
            local_mean,
            local_std,
            quat_mean,
            quat_std,
            global_mean,
            global_std,
            tgtanim,
            tgtname,
            tgtftime,
            inpanim,
            inpname,
            inpftime,
            inpjoints,
            tgtjoints,
        ) = load_from_preprocessed_mixamo(
            device=arg.device[0],
            src_char=src_char,
            motion_name=motion_name,
            num_joint=arg.num_joint,
            inp_shape_dir_path=arg.load_inp_data["inp_shape_dir_path"],
            tgt_shape_path=arg.load_inp_data["tgt_shape_path"],
            stats_path=arg.load_inp_data["stats_path"],
            inp_q_dir_path=arg.load_inp_data["inp_q_dir_path"],
        )

            # tgtskel from t-pose
            if tgtskel.shape[1] != inpskel.shape[1]:
                tgtskel = tgtskel[:, :1].repeat(1, inpskel.shape[1], 1)

            with torch.no_grad():
                localB_rt, globalB_rt, quatB_rt, delta_qs, delta_qg = ret_model(
                    inp_seq,
                    None,       # seqB: inference -> None
                    inpskel,
                    tgtskel,
                    inp_shape,
                    tgt_shape,
                    inpquat,
                    inp_height,
                    tgt_height,
                    local_mean,
                    local_std,
                    quat_mean,
                    quat_std,
                    parents,
                    arg.k,
                    arg.phase,
                )

            delta_qs = delta_qs.cpu()
            delta_qg = delta_qg.cpu()
            quatB_rt = quatB_rt.cpu()
            np.savez(save_path, delta_qs=delta_qs, delta_qg=delta_qg, quatB_rt=quatB_rt)

            # log
            processed_steps += 1
            elapsed = time.time() - start_time
            # format: R2ET_PROGRESS cur total elapsed
            print(
                f"R2ET_PROGRESS {processed_steps} {total_steps} {elapsed:.6f}",
                flush=True,
            )


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert k in key
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()

    if arg.job_id:
        # ./service/jobs/<job_id>
        job_root = os.path.join('.', 'service', 'jobs', arg.job_id)

        # 1) save_path: ./service/jobs/<job_id>/r2et
        arg.save_path = os.path.join(job_root, 'r2et')
        os.makedirs(arg.save_path, exist_ok=True)

        # 2) load_inp_data dict 준비
        if not hasattr(arg, 'load_inp_data') or arg.load_inp_data is None:
            arg.load_inp_data = {}

        # 3) tgt_shape_path / tgt_bvh_path overwrite
        arg.load_inp_data['tgt_shape_path'] = os.path.join(job_root, 'input.npz')
        arg.load_inp_data['tgt_q_path']   = job_root

    main(arg)
