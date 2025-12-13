# distilR²ET Demo 사용 방법

본 문서는 `distilr2et.onnx` 모델 파일을 이용하여  
Unity 환경에서 **Motion Retargeting 데모를 실행하는 방법**을 설명합니다.

---

## 1. Demo Scene 불러오기

1. distilR²ET Unity **플러그인 패키지**를 프로젝트에 임포트합니다.
2. 플러그인에 포함된 **DemoScene**을 엽니다.

DemoScene에는 Motion Retargeting 테스트를 위한  
기본적인 오브젝트 구성이 이미 포함되어 있습니다.

---

## 2. 모델 파일 (`distilr2et.onnx`) 설정

1. DemoScene에서 **Model 오브젝트**를 선택합니다.
2. 해당 오브젝트에 부착된  
   **`R2ET Retarget Sentis` 컴포넌트**를 찾습니다.
3. `Model Asset` 항목에  
   서버에서 다운로드한 **`distilr2et.onnx` 파일**을 할당합니다.

이 과정을 통해 Student 모델이 실제 추론에 사용됩니다.

---

## 3. 캐릭터 변경 방법 (선택)

기본 DemoScene에서는 **Claire 캐릭터**가 기본 Source Character로 설정되어 있습니다.  
다른 캐릭터를 사용하고 싶다면 아래 과정을 따르세요.

1. 원하는 캐릭터를 씬(Scene)에 불러옵니다.
2. **Model 오브젝트**를 선택합니다.
3. `Source Quat Reader Auto` 컴포넌트를 찾습니다.
4. `Character` 항목에  
   새로 불러온 **캐릭터 오브젝트**를 할당합니다.

이제 해당 캐릭터의 모션이 Retargeting 대상이 됩니다.

---

## 4. 애니메이션 설정

애니메이션 설정은 **일반적인 Unity 방식과 동일**합니다.

- 캐릭터 오브젝트의 **`Animator` 컴포넌트**를 통해
  - Animation Clip 변경
  - Runtime Animator Controller 교체
  - 파라미터 조정
  등을 수행할 수 있습니다.

Motion Retargeting을 위해 별도의 추가 설정은 필요하지 않습니다.

---

## 5. 오류 발생 시 확인 사항

데모 실행 중 문제가 발생한다면,  
대부분의 경우 **캐릭터 또는 애니메이션 설정 문제**일 가능성이 높습니다.

다음 항목을 우선적으로 확인하세요.

- 캐릭터의 **Avatar 설정 (Humanoid 여부)**
- 애니메이션 클립의 **Rig 설정**
- Animator Controller 연결 여부
- Source / Target 캐릭터의 스켈레톤 구조 적합성

오류 메시지를 기준으로  
Unity Animator 및 Rig 설정 방향으로 검색하는 것을 권장합니다.

---

## 6. 추가 플러그인 정보

distilR²ET Unity 플러그인에 대한  
자세한 구조, 컴포넌트 설명, 예제는 아래 저장소를 참고하세요.

> https://github.com/Kim-Hee-jae/distilR2ET_plugin

---

본 문서는 서버에서 전달되는 결과물 패키지에 포함되어  
사용자가 바로 데모를 실행할 수 있도록 돕는 용도입니다.
