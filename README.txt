1. 웹사이트로 받고
2. 전처리하고 <- 이때는 전처리 중 띄우고
3. 모델 학습 <- 모델 학습 예상 완료 시간 받아서 띄우고
4. onnx 빌드
5. tutorial 파일과 압축해서 다운


(Unity 메뉴 → Window → Package Manager → Unity Registry → “Editor Coroutines”).



uvicorn service.server:app --reload --host 0.0.0.0 --port 8000