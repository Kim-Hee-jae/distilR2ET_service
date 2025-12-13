# -*- coding: cp949 -*- 
# server.py
import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
from export_unity_onnx import export_onnx_for_job

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import zipfile
import subprocess
import sys, os

app = FastAPI()

# === 경로 설정 ===
# cd = distilR2ET
SERVICE_DIR = Path(__file__).resolve().parent          # .../distilR2ET/service
REPO_ROOT   = SERVICE_DIR.parent                       # .../distilR2ET
DATASETS_DIR = REPO_ROOT / "datasets"                  # .../distilR2ET/datasets

# 모든 작업은 ./service/jobs/<job_id>/ 안에서 처리
JOB_ROOT = SERVICE_DIR / "jobs"                        # .../distilR2ET/service/jobs
JOB_ROOT.mkdir(parents=True, exist_ok=True)

# Blender 실행 파일 경로
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 2.93\blender.exe"

USE_TMP_DATA_FOR_STUDENT = False # + should set config phase as 'tmp'


# === Job 상태 모델 ===
class JobStatus(BaseModel):
    job_id: str
    filename: str
    status: str           # "queued", "running", "done", "error"
    message: str = ""

    # 각 단계별 ETA (초)
    r2et_eta_seconds: int = -1      # -1: 아직 시작 안 함 or 사용 안 함
    student_eta_seconds: int = -1   # -1: 아직 시작 안 함 or 사용 안 함


jobs: Dict[str, JobStatus] = {}

# Job 취소 플래그 / 프로세스 / 스레드 관리
job_cancel_flags: Dict[str, bool] = {}
job_processes: Dict[str, List[subprocess.Popen]] = {}
job_threads: Dict[str, threading.Thread] = {}


def is_job_cancelled(job_id: str) -> bool:
    """해당 job_id가 취소 요청된 상태인지 여부"""
    return job_cancel_flags.get(job_id, False)


def register_proc(job_id: str, proc: subprocess.Popen) -> None:
    """job_id에 속한 하위 프로세스를 등록"""
    job_processes.setdefault(job_id, []).append(proc)


def kill_job_processes(job_id: str) -> None:
    """job_id와 연결된 모든 하위 프로세스를 kill"""
    procs = job_processes.get(job_id, [])
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    job_processes[job_id] = []


def mark_job_cancelled(job_id: str) -> None:
    """취소 플래그 켜고 프로세스 전부 kill"""
    job_cancel_flags[job_id] = True
    kill_job_processes(job_id)



# === 전처리 파이프라인 ===
def run_preprocess_pipeline(job_id: str, job_dir: Path, log_prefix: Optional[str] = None) -> None:
    """
    job_dir 안에 저장된 input.fbx에 대해
    ./datasets/fbx2bvh.py, preprocess_q.py, extract_shape.py 를 순차 실행

    - job_dir : service/jobs/<job_id>/ 디렉토리 (input.fbx 필요)
    - log_prefix : 로그 메시지 앞에 붙일 문자열 (선택 사항)
    """

    if log_prefix is None:
        log_prefix = "[PREPROCESS]"

    input_fbx = job_dir / "input.fbx"
    if not input_fbx.exists():
        raise FileNotFoundError(f"{log_prefix} input.fbx 가 존재하지 않습니다: {input_fbx}")

    if not DATASETS_DIR.exists():
        raise FileNotFoundError(f"{log_prefix} DATASETS_DIR 를 찾을 수 없습니다: {DATASETS_DIR}")

    # 실행할 전처리 커맨드
    commands = [
        # 1) fbx2bvh.py : Blender 기반, job_dir 인자 전달
        [
            BLENDER_EXE,
            "-b",
            "-P", str(DATASETS_DIR / "fbx2bvh.py"),
            "--",
            str(job_dir),
        ],
        # 2) preprocess_q.py : 일반 Python, job_dir 인자 전달
        [
            sys.executable,
            str(DATASETS_DIR / "preprocess_q.py"),
            str(job_dir),
        ],
        # 3) extract_shape.py : Blender 기반, job_dir 인자 전달
        [
            BLENDER_EXE,
            "-b",
            "-P", str(DATASETS_DIR / "extract_shape.py"),
            "--",
            str(job_dir),
        ],
    ]

    for cmd in commands:
        if is_job_cancelled(job_id):
            print(f"{log_prefix} 취소 플래그 감지, 전처리 중단")
            raise RuntimeError("Job cancelled during preprocess")

        print(f"{log_prefix} 실행: {' '.join(cmd)} (cwd={REPO_ROOT})")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            register_proc(job_id, proc)

            stdout_bytes, stderr_bytes = proc.communicate()

            stdout = stdout_bytes.decode("utf-8", errors="ignore") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="ignore") if stderr_bytes else ""

            if stdout:
                print(f"{log_prefix} STDOUT:\n{stdout}")
            if stderr:
                print(f"{log_prefix} STDERR:\n{stderr}")

            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, cmd, stdout_bytes, stderr_bytes
                )

        except subprocess.CalledProcessError as e:
            print(f"{log_prefix} 명령 실행 실패: {' '.join(cmd)}")
            print(f"{log_prefix} 반환 코드: {e.returncode}")

            if e.stdout:
                stdout = e.stdout.decode("utf-8", errors="ignore")
                print(f"{log_prefix} STDOUT:\n{stdout}")
            if e.stderr:
                stderr = e.stderr.decode("utf-8", errors="ignore")
                print(f"{log_prefix} STDERR:\n{stderr}")

            raise


# === R2ET 추론 ===
def run_r2et_inference(job_id: str):
    job_dir = JOB_ROOT / job_id

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "inference_r2et2.py"),
        "--config", str(REPO_ROOT / "config" / "inference_r2et2.yaml"),
        "--job_id", job_id,
    ]

    print(f"[JOB {job_id}] R2ET 명령 실행: {' '.join(cmd)} (cwd={REPO_ROOT})")

    total_steps = None 

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # stdout + stderr 합침
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    register_proc(job_id, proc)

    try:
        # 시작할 때 ETA 초기화
        job = jobs.get(job_id)
        if job:
            job.r2et_eta_seconds = -1
            job.message = "학습 데이터 제작 시작"
            jobs[job_id] = job
            
        # child 의 stdout 줄 단위 실시간 파싱
        for raw_line in proc.stdout:
            if raw_line is None:
                break

            if is_job_cancelled(job_id):
                print(f"[JOB {job_id}] 취소 플래그 감지, R2ET 프로세스 kill")
                try:
                    proc.kill()
                except Exception:
                    pass
                raise RuntimeError("Job cancelled during R2ET inference")

            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            print(f"[JOB {job_id}] R2ET OUT: {line}")

            # 총 스텝 수
            if line.startswith("R2ET_TOTAL_STEPS"):
                try:
                    _, n_str = line.split()
                    total_steps = int(n_str)
                    print(f"[JOB {job_id}] total_steps = {total_steps}")
                except Exception as e:
                    print(f"[JOB {job_id}] total_steps 파싱 실패: {e}")

            # 진행 상황 + ETA 계산
            elif line.startswith("R2ET_PROGRESS") and total_steps:
                try:
                    # 형식: R2ET_PROGRESS cur total elapsed
                    _, cur_str, total_str, elapsed_str = line.split()
                    cur_step = int(cur_str)
                    total = int(total_str)
                    elapsed = float(elapsed_str)

                    avg_per_step = elapsed / max(cur_step, 1)
                    remain_steps = max(total - cur_step, 0)
                    eta_sec = int(avg_per_step * remain_steps)

                    job = jobs.get(job_id)
                    if job:
                        job.r2et_eta_seconds = max(eta_sec, 0)
                        job.message = f"학습 데이터 제작 중... ({cur_step}/{total})"
                        jobs[job_id] = job

                except Exception as e:
                    print(f"[JOB {job_id}] R2ET_PROGRESS 파싱 실패: {e}")

        ret = proc.wait()

        if ret != 0:
            raise RuntimeError(f"R2ET inference failed with code {ret}")

        # 끝나면 ETA 0으로
        job = jobs.get(job_id)
        if job:
            job.r2et_eta_seconds = 0
            jobs[job_id] = job

        print(f"[JOB {job_id}] run_r2et_inference 정상 종료")

    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        print(f"[JOB {job_id}] run_r2et_inference 예외: {e}")
        raise

# === Student 학습 ===
def run_student_training(job_id: str, data_job_id: Optional[str] = None):
    """
    job_id:     UI/서버에서 관리하는 jobId (ETA 갱신 대상)
    data_job_id: train_student.py에 넘길 --job_id (데이터/경로용)
                 None이면 job_id와 동일하게 사용
    """
    train_job_id = data_job_id or job_id

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "train_student.py"),
        "--config", str(REPO_ROOT / "config" / "train_student.yaml"),
        "--job_id", train_job_id,
    ]

    print(f"[JOB {job_id}] STUDENT 명령 실행: {' '.join(cmd)} (cwd={REPO_ROOT}, data_job_id={train_job_id})")

    total_steps = None  

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True, 
    )

    register_proc(job_id, proc)

    try:
        # 시작할 때 ETA 초기화
        job = jobs.get(job_id)
        if job:
            job.student_eta_seconds = -1
            job.message = "모델 학습 시작"
            jobs[job_id] = job

        for line in proc.stdout:
            if is_job_cancelled(job_id):
                print(f"[JOB {job_id}] 취소 플래그 감지, STUDENT 프로세스 kill")
                try:
                    proc.kill()
                except Exception:
                    pass
                raise RuntimeError("Job cancelled during student training")

            line = line.strip()
            if not line:
                continue

            print(f"[JOB {job_id}] STUDENT OUT: {line}")

            # 총 스텝 수
            if line.startswith("STUDENT_TOTAL_STEPS"):
                try:
                    _, n_str = line.split()
                    total_steps = int(n_str)
                    print(f"[JOB {job_id}] STUDENT total_steps = {total_steps}")
                except Exception as e:
                    print(f"[JOB {job_id}] STUDENT total_steps 파싱 실패: {e}")

            # 진행 상황 + ETA 계산
            elif line.startswith("STUDENT_PROGRESS") and total_steps:
                try:
                    # 형식: STUDENT_PROGRESS cur total elapsed
                    _, cur_str, total_str, eta_sec = line.split()
                    cur_step = int(cur_str)
                    total = int(total_str)
                    eta_sec = float(eta_sec)

                    job = jobs.get(job_id)
                    if job:
                        # student 단계 ETA
                        job.student_eta_seconds = max(eta_sec, 0)
                        job.message = f"모델 학습 중... ({cur_step}/{total})"
                        jobs[job_id] = job

                except Exception as e:
                    print(f"[JOB {job_id}] STUDENT_PROGRESS 파싱 실패: {e}")

        # 프로세스 종료 대기
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"train_student.py 비정상 종료 (returncode={proc.returncode})")

        print(f"[JOB {job_id}] run_student_training 정상 종료")

    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        print(f"[JOB {job_id}] run_student_training 예외: {e}")
        raise


# === ONNX Export===
def run_onnx_export(job_id: str):
    """
    학습된 student 모델을 ONNX 파일로 내보내는 단계
    """
    if is_job_cancelled(job_id):
        print(f"[JOB {job_id}] ONNX export 시작 전에 취소 플래그 감지, 스킵")
        raise RuntimeError("Job cancelled before ONNX export")

    # 상태 메시지 갱신
    job = jobs.get(job_id)
    if job:
        job.message = "모델 업로드 중"
        jobs[job_id] = job

    try:
        print(f"[JOB {job_id}] ONNX export 시작")
        export_onnx_for_job(
            job_id=job_id,
            jobs_root=str(JOB_ROOT),   # 기존과 동일
        )
        print(f"[JOB {job_id}] ONNX export 완료")

    except Exception as e:
        print(f"[JOB {job_id}] run_onnx_export 예외: {e}")
        # run_training_job 쪽에서 잡아서 status='error' 설정
        raise

    # ONNX export가 끝난 뒤에라도 취소 요청이 들어갔을 수 있으니 한 번 더 체크
    if is_job_cancelled(job_id):
        print(f"[JOB {job_id}] ONNX export 직후 취소 플래그 감지, 이후 단계 스킵")
        raise RuntimeError("Job cancelled after ONNX export")


# === 학습 + ZIP 생성 ===
def run_training_job(job_id: str):
    if is_job_cancelled(job_id):
        print(f"[JOB {job_id}] 이미 취소된 작업, 즉시 종료")
        return

    job = jobs[job_id]
    job.status = "running"
    job.message = "전처리 중..."
    jobs[job_id] = job

    job_dir = JOB_ROOT / job_id
    input_path = job_dir / "input.fbx"
    result_zip = job_dir / "result.zip"
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        print(f"[JOB {job_id}] ===== 전처리 시작 =====")
        # 1) 전처리 (fbx2bvh, preprocess_q, extract_shape)
        run_preprocess_pipeline(job_id, job_dir, log_prefix=f"[JOB {job_id}]")
        print(f"[JOB {job_id}] ===== 전처리 완료 =====")
        if is_job_cancelled(job_id):
            print(f"[JOB {job_id}] 전처리 후 취소됨, 나머지 단계 스킵")
            return

        print(f"[JOB {job_id}] ===== R2ET inference 시작 =====")
        # 2) 전처리된 데이터를 기반으로 실제 모델 추론 수행
        # 테스트 모드
        if not USE_TMP_DATA_FOR_STUDENT:
            run_r2et_inference(job_id)
            print(f"[JOB {job_id}] ===== R2ET inference 완료 =====")
        else:
            print(f"[JOB {job_id}] ===== 테스트 모드: R2ET / 전처리 스킵, tmp 데이터로 student 학습만 진행 =====")
        if is_job_cancelled(job_id):
            print(f"[JOB {job_id}] 전처리 후 취소됨, 나머지 단계 스킵")
            return
        
        print(f"[JOB {job_id}] ===== Student 학습 시작 =====")
        # 3) teacher 결과를 기반으로 student distillation 학습
        run_student_training(job_id=job_id, data_job_id=job_id)
        print(f"[JOB {job_id}] ===== Student 학습 완료 =====")
        if is_job_cancelled(job_id):
            print(f"[JOB {job_id}] 전처리 후 취소됨, 나머지 단계 스킵")
            return

        # 4) 학습된 모델 onnx 파일로 변환
        if is_job_cancelled(job_id):
            print(f"[JOB {job_id}] Student 학습 후 취소됨, ONNX export 스킵")
            return

        run_onnx_export(job_id=job_id)

        if is_job_cancelled(job_id):
            print(f"[JOB {job_id}] ONNX export 후 취소됨, ZIP 생성 스킵")
            return

        # 사용법 README
        service_readme_path = Path("./service/README.md")
        shutil.copy(service_readme_path, output_dir / "README.md")

        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in output_dir.rglob("*"):
                zf.write(p, p.relative_to(job_dir))

        job.status = "done"
        job.message = "완료"
        jobs[job_id] = job

    except Exception as e:
        print(f"[JOB {job_id}] run_training_job 중 에러 발생: {e}")
        job.status = "error"
        job.message = f"에러: {e}"
        jobs[job_id] = job



# === 업로드 엔드포인트 ===
@app.post("/api/upload")
async def upload_fbx(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".fbx"):
        raise HTTPException(status_code=400, detail="FBX 파일만 허용됩니다.")

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 업로드된 FBX를 해당 job 폴더에 저장
    input_path = job_dir / "input.fbx"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = JobStatus(
        job_id=job_id,
        filename=file.filename,
        status="queued",
        message="대기 중",
        )
    jobs[job_id] = job

    job_cancel_flags[job_id] = False
    job_processes[job_id] = []

    t = threading.Thread(target=run_training_job, args=(job_id,), daemon=True)
    job_threads[job_id] = t
    t.start()

    return {
        "jobId": job_id,
        "status": job.status,
    }


# === Job 삭제 ===
@app.delete("/api/job/{job_id}")
def delete_job(job_id: str):
    # 0) 취소 플래그 켜고, 관련 프로세스 모두 kill
    if job_id in jobs:
        print(f"[JOB {job_id}] delete 요청 -> 취소 플래그 ON + 프로세스 kill")
        mark_job_cancelled(job_id)

    # 1) 백그라운드 스레드가 있다면 잠깐 기다려서 정리
    t = job_threads.get(job_id)
    if t and t.is_alive():
        # timeout 부여
        t.join(timeout=5.0)

    # 2) 메모리 상 job 상태 제거
    if job_id in jobs:
        del jobs[job_id]
    if job_id in job_threads:
        del job_threads[job_id]
    if job_id in job_processes:
        del job_processes[job_id]
    if job_id in job_cancel_flags:
        del job_cancel_flags[job_id]

    # 3) 폴더 삭제
    job_dir = JOB_ROOT / job_id
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"폴더 삭제 실패: {e!r}")

    return {"ok": True, "jobId": job_id}



# === 상태 조회 ===
@app.get("/api/status")
def get_status(jobId: str):
    job = jobs.get(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="존재하지 않는 jobId 입니다.")
    return {
        "jobId": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "message": job.message,
        "r2etEtaSeconds": job.r2et_eta_seconds,
        "studentEtaSeconds": job.student_eta_seconds,
    }


# === 결과 ZIP 다운로드 ===
@app.get("/api/download")
def download_result(jobId: str):
    job = jobs.get(jobId)
    if not job:
        raise HTTPException(status_code=404, detail="존재하지 않는 jobId 입니다.")

    if job.status != "done":
        raise HTTPException(status_code=400, detail="아직 완료되지 않은 작업입니다.")

    job_dir = JOB_ROOT / jobId
    result_zip = job_dir / "result.zip"

    if not result_zip.exists():
        raise HTTPException(status_code=500, detail="결과 파일이 없습니다.")

    return FileResponse(
        path=result_zip,
        filename=f"job-{jobId}-result.zip",
        media_type="application/zip",
    )
