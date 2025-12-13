# -*- coding: cp949 -*-
import json
import random
import os
from pathlib import Path
import sys 
import yaml

def make_random_split(
    dataset_dir: str,
    unseen_ratio: float,
    out_path: str,
    seed: int = 42,
    exts=None,
    seen_char=None,
    unseen_char=None,
    target_char: str = "Claire",
):
    """
    dataset_dir 안의 파일들 중 unseen_ratio 를 랜덤으로 뽑아서
    unseen / train 리스트를 out_path(json)에 저장
    """
    dataset_dir = Path(dataset_dir)
    assert dataset_dir.is_dir(), f"{dataset_dir} is not a directory"

    # 파일 모으기
    if exts is None:
        files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    else:
        exts = {e.lower() for e in exts}
        files = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in exts]

    files = sorted(files)
    n_total = len(files)
    if n_total == 0:
        raise ValueError(f"No files found in {dataset_dir}")

    # 랜덤 셔플 후 n%를 unseen으로
    random.seed(seed)
    random.shuffle(files)

    n_unseen = int(n_total * unseen_ratio)
    unseen_files = files[:n_unseen]
    train_files = files[n_unseen:]

    # 기본값 처리
    if seen_char is None:
        seen_char = []
    if unseen_char is None:
        unseen_char = []

    result = {
        "dataset_dir": str(dataset_dir),
        "total_files": n_total,
        "unseen_ratio": unseen_ratio,
        "n_seen": len(train_files),
        "n_unseen": len(unseen_files),
        "seed": seed,
        "seen_motion": [str(p) for p in train_files],
        "unseen_motion": [str(p) for p in unseen_files],
        "seen_char": seen_char,
        "unseen_char": unseen_char,
        "target_char": target_char,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved split to {out_path}")
    print(f"  total  : {n_total}")
    print(f"  seen   : {len(train_files)}")
    print(f"  unseen : {len(unseen_files)}")
    print(f"  seen_char   : {seen_char}")
    print(f"  unseen_char : {unseen_char}")
    print(f"  target_char : {target_char}")

def update_inference_configs(target_char: str, config_dir: str = "./config"):
    """
    ./config/inference_all_bvh.yaml
    ./config/inference_all_bvh_student.yaml

    두 파일을 target_char에 맞춰 변경:
      load_inp_data:
        tgt_shape_path: ./datasets/mixamo/shape/Claire.npz
          -> ./datasets/mixamo/shape/{target_char}.npz

        tgt_bvh_path: ./datasets/mixamo/char/Claire/
          -> ./datasets/mixamo/char/{target_char}/
    """
    config_dir_path = Path(config_dir)
    filenames = ["inference_all_bvh.yaml", "inference_all_bvh_student.yaml"]

    for name in filenames:
        cfg_path = config_dir_path / name
        if not cfg_path.exists():
            print(f"[Config] WARN: {cfg_path} not found, skip")
            continue

        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            print(f"[Config] WARN: {cfg_path} is not a dict-like yaml, skip")
            continue

        # load_inp_data 내부 수정
        lid = cfg.get("load_inp_data", {})
        if isinstance(lid, dict):
            lid["tgt_shape_path"] = f"./datasets/mixamo/shape/{target_char}.npz"
            lid["tgt_bvh_path"] = f"./datasets/mixamo/char/{target_char}/"
            cfg["load_inp_data"] = lid
        else:
            print(f"[Config] WARN: 'load_inp_data' in {cfg_path} is not a dict, skip that part")

        # 다시 저장
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        print(f"[Config] updated: {cfg_path}")


if __name__ == "__main__":
    """
    사용법 예시:

      # 1) unseen_ratio만 지정
      #    unseen_ratio = 0.2, unseen_char_count = 1, target_char = "Claire"
      python ./datasets/data_splitter.py 0.2

      # 2) unseen_ratio, unseen_char 개수 지정
      #    unseen_ratio = 0.3, unseen_char_count = 2, target_char = "Claire"
      python ./datasets/data_splitter.py 0.3 2

      # 3) unseen_ratio, unseen_char 개수, target_char까지 지정
      #    unseen_ratio = 0.25, unseen_char_count = 3, target_char = "Steve"
      python ./datasets/data_splitter.py 0.25 3 Steve
    """

    # 1) unseen_ratio
    if len(sys.argv) >= 2:
        unseen_ratio = float(sys.argv[1])
    else:
        unseen_ratio = 0.20  # 기본값

    # 2) unseen_char 개수 
    if len(sys.argv) >= 3:
        unseen_char_count = int(sys.argv[2])
    else:
        unseen_char_count = 1 # 기본값

    # 3) target_char
    if len(sys.argv) >= 4:
        target_char = sys.argv[3]
    else:
        target_char = "Claire" # 기본값

    # 캐릭터 목룍
    char_root = "./datasets/mixamo/char/"
    all_entries = os.listdir(char_root)
    all_chars = sorted(
        [
            d for d in all_entries
            if os.path.isdir(os.path.join(char_root, d))
        ]
    )

    if not all_chars:
        raise RuntimeError(f"No character folders found in {char_root}")

    # target_char는 seen/unseen 분리에서 제외
    chars_no_target = [c for c in all_chars if c != target_char]

    if not chars_no_target:
        # 모든 캐릭터가 target_char인 경우 -> 경고
        print(f"[WARNING] only target_char '{target_char}' found. seen/unseen_char will be empty.")
        seen_char = []
        unseen_char = []
    else:
        # unseen_char 개수
        unseen_char_count = max(0, min(unseen_char_count, len(chars_no_target)))

        # 뒤에서 unseen_char_count개 사용
        unseen_char = chars_no_target[-unseen_char_count:]

        # random_version
        # random.seed(42)
        # unseen_char = random.sample(chars_no_target, unseen_char_count)

        seen_char = [c for c in chars_no_target if c not in unseen_char]

    # dataset_dir 설정: target_char 폴더 기준으로 split
    folder_name = target_char 
    dataset_dir = os.path.join(char_root, folder_name)

    # 출력 경로
    out_path = "./config/data_config.json"

    make_random_split(
        dataset_dir=dataset_dir,
        unseen_ratio=unseen_ratio,
        out_path=out_path,
        seed=42,
        exts=[".bvh"], 
        seen_char=seen_char,
        unseen_char=unseen_char,
        target_char=target_char,
    )

    update_inference_configs(target_char=target_char, config_dir="./config")
