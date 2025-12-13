# -*- coding: cp949 -*-
# export_unity_onnx.py
#
# 사용 예시:
#   python export_unity_onnx.py --job_id 20251213_123456 \
#       --jobs_root ./service/jobs
#
# 결과:
#   ./service/jobs/[job_id]/output/distilr2et.onnx 생성

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from src.model_student_unity import DistilledRetNet, UnityRetargetWrapper


def _strip_module_prefix(state_dict: dict) -> dict:
    """
    DataParallel 로 학습된 모델의 state_dict 에서 'module.' prefix 제거.
    """
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    return new_state


def load_constants_for_job(job_dir: Path):
    """
    Unity 추론에 필요한 상수들 로드
    - stats: service/jobs/[job_id]/r2et/stats/mixamo_*.npy
    - target skeleton: service/jobs/[job_id]/input_skel.npy
    """
    stats_dir = job_dir / "r2et" / "stats"

    local_mean = np.load(stats_dir / "mixamo_local_motion_mean.npy")
    local_std  = np.load(stats_dir / "mixamo_local_motion_std.npy")
    quat_mean  = np.load(stats_dir / "mixamo_quat_mean.npy")
    quat_std   = np.load(stats_dir / "mixamo_quat_std.npy")

    # Mixamo 22-joint skeleton 트리 (고정)
    parents = np.array(
        [-1, 0, 1, 2, 3, 4,
          0, 6, 7, 8,
          0, 10,11,12,
          3, 14,15,16,
          3, 18,19,20]
    )

    # target skeleton 파일
    tgt_skel_path = job_dir / "input_skel.npy"
    if not tgt_skel_path.exists():
        raise FileNotFoundError(f"Target skeleton not found: {tgt_skel_path}")

    tgt_skel = np.load(tgt_skel_path)  # (J,3) 혹은 (1,J,3) 가정
    tgt_height = DistilledRetNet.get_height_from_skel(tgt_skel[:1])  # (1,)

    return local_mean, local_std, quat_mean, quat_std, parents, tgt_height[np.newaxis]


def build_model_for_job(job_dir: Path, ckpt_path: Optional[Path] = None):
    """
    DistilledRetNet + UnityRetargetWrapper 생성 및 weight 로드
    """
    # 1) base DistilledRetNet 생성
    base_model = DistilledRetNet(
        num_joint=22,
        token_channels=64,
        hidden_channels_p=64,
        embed_channels_p=64,
        kp=1.0,
        limited_input=True,   # 실시간일 때 스켈 임베딩 1번만 계산
    )

    # 2) checkpoint 경로 결정
    if ckpt_path is None:
        ckpt_path = job_dir / "student/distilled_r2et.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[export_unity_onnx] Using checkpoint: {ckpt_path}")

    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = _strip_module_prefix(state_dict)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    # 4) 상수 로드
    local_mean, local_std, quat_mean, quat_std, parents, tgt_height = load_constants_for_job(job_dir)

    # 5) Unity용 래퍼 생성
    wrapper = UnityRetargetWrapper(
        base_model=base_model,
        tgt_height=tgt_height,
        local_mean=local_mean,
        local_std=local_std,
        quat_mean=quat_mean,
        quat_std=quat_std,
        parents=parents,
        k=-1.0,
        phase="inference",
    )

    return wrapper


def export_onnx_for_job(
    job_id: str,
    jobs_root: str = "./service/jobs",
):
    """
      - ./service/jobs/[job_id]/student/distilled_r2et/ 의 최신 .pt 사용
      - ./service/jobs/[job_id]/r2et/stats/ 의 mixamo_*.npy 사용
      - ./service/jobs/[job_id]/input_skel.npy 사용
      - ./service/jobs/[job_id]/output/distilr2et.onnx 로 export
    """
    jobs_root = Path(jobs_root)
    job_dir = jobs_root / job_id

    if not job_dir.exists():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")

    # 1) 모델 구성
    model = build_model_for_job(job_dir)
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    # 2) 더미 입력 생성 (B=1, T=1 기준)
    B = 1
    T = 1
    J = 22

    # seqA : (B, T, J*3 + 4)  -> root vel (J*3) + root rot_y (1) + 기타 (3) = 70
    seqA_dummy = torch.zeros(B, T, J * 3 + 4, dtype=torch.float32, device=device)    # (1,1,70)

    # quatA : (B, T, J, 4)
    quatA_dummy = torch.zeros(B, T, J, 4, dtype=torch.float32, device=device)        # (1,1,22,4)

    # skelA : (B, T, J, 3)
    skelA_dummy = torch.zeros(B, T, J, 3, dtype=torch.float32, device=device)        # (1,1,22,3)

    # shapeA : (B, J*3)
    shapeA_dummy = torch.zeros(B, J * 3, dtype=torch.float32, device=device)         # (1,66)

    # inp_height : (B, 1)
    inp_height_dummy = torch.ones(B, 1, dtype=torch.float32, device=device) * 1.7    # (1,1)

    dummy_inputs = (seqA_dummy, quatA_dummy, skelA_dummy, shapeA_dummy, inp_height_dummy)

    # 3) 출력 경로 결정
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_output_path = output_dir / "distilr2et.onnx"

    # 4) ONNX export
    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            "seqA",        # (B,T,J*3+4)
            "quatA",       # (B,T,J,4)
            "skelA",       # (B,T,J,3)
            "shapeA",      # (B,J*3)
            "inp_height",  # (B,1)
        ],
        output_names=[
            "globalB",     # (B,T,4)
            "quatB",       # (B,T,J,4)
        ],
        dynamic_axes={
            "seqA":      {0: "batch", 1: "time"},
            "quatA":     {0: "batch", 1: "time"},
            "skelA":     {0: "batch", 1: "time"},
            "shapeA":    {0: "batch"},
            "inp_height":{0: "batch"},
            "globalB":   {0: "batch", 1: "time"},
            "quatB":     {0: "batch", 1: "time"},
        },
    )

    print(f"[export_unity_onnx] ONNX model exported to: {onnx_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_id",
        type=str,
        required=True,
        help="service/jobs 아래의 job 폴더 이름",
    )
    parser.add_argument(
        "--jobs_root",
        type=str,
        default="./service/jobs",
        help="job 들이 위치한 루트 폴더 (기본: ./service/jobs)",
    )
    args = parser.parse_args()

    export_onnx_for_job(
        job_id=args.job_id,
        jobs_root=args.jobs_root,
    )
