# distilR²ET Service
**Distillated Motion Retargeting Service Backend**

---

## 개요 (Overview)

본 저장소는 **distilR²ET Motion Retargeting 프레임워크의 서비스(Server) 코드**를 포함합니다.  
사용자로부터 **FBX 파일을 입력으로 받아**, 전처리부터 학습, ONNX 모델 생성까지 자동으로 수행한 뒤  
결과물을 **압축하여 반환**하는 서버 애플리케이션입니다.

본 저장소는 연구/개발 목적의 `distilR²ET_develop` 저장소와 분리된 **서비스 전용 코드**입니다.
사용하는 가상환경은 `distilR²ET_develop`와 동일하니 서비스 코드 사용 전에  
https://github.com/Kim-Hee-jae/distilR2ET_develop  
를 참고하여 가상환경부터 설치하세요.

---

## 프로젝트 폴더 구조

```text
config/          - 서버 및 모델 설정 파일
datasets/        - 서비스 중 사용되는 데이터셋 !용량 문제로 인해 메일로 전달!
outside-code/    - FBX/BVH 처리 관련 외부 코드
pretrain/        - 사전학습된 R²ET Teacher (checkpoint)
src/             - 모델 구조 및 기타 코드
service/         - 서비스 입출력 및 서버 코드 (storage 역할)
```

- `service/` 폴더는 서비스 실행 중 **입출력 파일이 저장되는 공간**으로,
  서버의 **storage 디렉토리** 역할을 합니다.
- 나머지 폴더 구조는 서비스 중 동일하게 유지되어,
  **학습·추론 파이프라인의 일관성**을 보장합니다.

---

## 전체 서비스 파이프라인 한눈에 보기

```text
Client
  │
  │  (FBX 업로드)
  ▼
Server
  ├─ FBX 파일 수신
  ├─ FBX → BVH 변환 및 전처리
  ├─ 학습용 데이터 생성
  ├─ distilR²ET Student 모델 학습 및 평가
  ├─ ONNX 모델 변환
  └─ 결과물 압축 후 송신
```

---

## 서버 실행 방법 (Quick Start)

### 0. 가상환경 설정

참고: https://github.com/Kim-Hee-jae/distilR2ET_develop

### 1. ngrok 설정 및 실행

외부에서 서버에 접근하기 위해 **ngrok**을 사용합니다.

#### 1-1. ngrok 다운로드 및 계정 생성

- https://dashboard.ngrok.com/get-started/setup/windows
- 회원가입 후 **Auth Token 발급**

#### 1-2. ngrok 인증 토큰 등록

```bash
ngrok config add-authtoken '본인_토큰'
```

#### 1-3. 포트 8000 노출

```bash
ngrok http 8000
```

> 실행 후 출력되는 **HTTPS URL**을 통해 외부 요청이 서버로 전달됩니다.

---

### 2. 서버 실행

```bash
uvicorn service.server:app --reload --host 0.0.0.0 --port 8000
```

- FastAPI + Uvicorn 기반 서버
- 기본 포트: **8000**

---

## 서버가 수행하는 작업

서버는 요청 1회당 다음 과정을 **자동으로 순차 수행**합니다.

1. **FBX 파일 수신**
2. **전처리**
   - FBX → BVH 변환
   - 모델 입력 포맷에 맞는 데이터 전처리 및 정규화 
3. **학습용 데이터 생성**
4. **distilR²ET Student 모델 학습**
5. **ONNX 모델 변환**
6. **결과물 패키징 및 송신**

---

## 반환 파일 설명

서버는 아래 파일들을 **압축(zip)** 하여 클라이언트로 반환합니다.

### 1. `distilr2et.onnx`

- 최종 학습된 **Student Motion Retargeting 모델**
- ONNX 포맷
- 실시간 추론 및 서비스 배포에 사용 가능

---

### 2. `generalization_gap.txt`

- Teacher(R²ET) 대비 Student 모델의 **일반화 성능 차이**를 기록한 파일
- 내용:
  - 대형 모델(teacher: R2ET) 대비 일반화 성능 차이
    평가 unseen 캐릭터 & unseen 애니메이션
    평균 quatB 차이 (teacher: R2ET): 9.67 ± 13.20 degrees


---

### 3. `README.md`

- 서버 요청을 통해 생성된 결과물에 대한 요약 및 사용 가이드 문서
- 포함 정보:
  - 모델 사용에 대한 간단한 안내
  - 오류 발생 시 대처 방안

---

## 주의 사항

- 본 저장소는 **서비스 실행을 위한 코드만 포함**합니다.
- 연구 실험 및 분석 목적의 코드는  
  `distilR²ET_develop` 저장소를 참고하십시오.
- `service/` 폴더는 서버 실행 중 **동적으로 변경**될 수 있습니다.
- 학습 과정이 포함되므로 요청에 따라 **처리 시간이 길어질 수 있습니다**.

---

## 관련 저장소

- **Develop Repository**  
  https://github.com/Kim-Hee-jae/distilR2ET_develop
