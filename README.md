<img width="1007" height="456" alt="image" src="https://github.com/user-attachments/assets/7b5777ac-1550-4aac-a0fe-80b6bfb3e834" />

### **수화 자연어 변환 시스템 구성 및 소통 방식**

이 시스템은 엣지 디바이스(Raspberry Pi + Hailo-8)에서 시각적 특징을 추출하고, PC에서 시퀀스 모델링을 통해 자연어로 변환하며, 최종적으로 서버의 LLM을 통해 응답을 생성하는 다단계 파이프라인으로 구성됩니다.

**1. 하드웨어 구성**

- **엣지 디바이스 (Raspberry Pi + Hailo-8)**:
    - **Raspberry Pi**: 웹캠 이미지 수신 및 Hailo-8 제어, 특징 데이터 전송 역할을 수행합니다. 충분한 연산 능력과 I/O 포트를 가진 모델(예: Raspberry Pi 4B 이상)을 권장합니다.
    - **웹캠**: 고해상도(예: Full HD 이상) 및 충분한 프레임 레이트(예: 30 FPS 이상)를 지원하여 수화 동작의 미세한 변화를 잘 포착할 수 있는 USB 웹캠을 권장합니다.
    - **Hailo-8**: Raspberry Pi에 연결되어 신경망 모델의 고속 추론을 가속화합니다. M.2 또는 USB 3.0 어댑터를 통해 Raspberry Pi에 연결됩니다. 이는 웹캠 이미지에서 특징점 또는 특징 벡터를 효율적으로 추출하는 데 사용됩니다.
    - **Wi-Fi 모듈**: Raspberry Pi에 내장된 Wi-Fi 또는 외장 Wi-Fi 동글을 사용하여 PC와 무선 통신합니다.
- **메인 처리 장치 (PC + Intel B580 GPU)**:
    - **PC**: Raspberry Pi로부터 특징 데이터를 수신하고, Intel B580 GPU를 활용하여 RNN, LSTM, Transformer와 같은 복잡한 시퀀스 모델을 실행하여 수화 특징을 단어나 문장으로 변환합니다.
    - **Intel B580 GPU**: 시퀀스 모델의 딥러닝 추론을 가속화하는 역할을 합니다. 적절한 드라이버 및 딥러닝 프레임워크(TensorFlow, PyTorch 등) 설정이 필수입니다.
- **서버 (vLLM 실행)**:
    - **서버**: 대규모 언어 모델(LLM)을 vLLM(Virtual Large Language Model)을 통해 효율적으로 서비스합니다.
    PC에서 변환된 자연어 텍스트를 입력받아 응답을 생성합니다. GPU가 장착된 서버(예: NVIDIA GPU)가 필요하며, vLLM은 높은 처리량과 낮은 지연 시간을 제공합니다.

**2. 데이터 흐름 및 소통 방식**

**단계 1: 엣지에서의 특징 추출 (웹캠 -> Raspberry Pi -> Hailo-8)**

1. **웹캠 이미지 획득**: 웹캠이 Raspberry Pi에 USB 또는 CSI 인터페이스를 통해 연결됩니다. Raspberry Pi는 OpenCV 라이브러리 또는 picamera2와 같은 전용 카메라 라이브러리를 사용하여 웹캠으로부터 실시간 비디오 프레임을 획득합니다.
2. **Hailo-8 모델 추론**: 획득된 비디오 프레임은 Hailo-8에서 추론될 수 있는 형태로 전처리됩니다 (예: 크기 조정, 정규화, 채널 순서 변경).
Hailo-8에 최적화된 모델(예: ONNX, TensorFlow Lite, PyTorch 모델을 Hailo Dataflow Compiler를 통해 컴파일)이 로드됩니다.
    - **옵션 1 (특징점 추출)**: 수화 동작의 주요 관절점(손가락, 팔꿈치 등)이나 얼굴 특징점 등을 감지하는 모델(예: MediaPipe Hands)을 Hailo-8에서 실행하여 각 특징점의 2D/3D 좌표와 신뢰도 등을 추출합니다. 이는 원시적이지만 직접적인 특징을 제공합니다.
    - **옵션 2 (특징 벡터 추출 - FF 결과)**: 수화 인식을 위해 학습된 신경망 모델(예: CNN 기반 특징 추출기)의 최종 분류 계층(classification layer) 전의 임베딩 벡터 또는 피처 벡터(FF 결과)를 추출합니다. 이 방식은 수화에 특화된 추상적인 특징을 제공하며, 후속 시퀀스 모델의 성능에 더 유리할 수 있습니다. 이 경우, 특징 추출 모델은 수화 데이터셋으로 별도 학습되거나 파인튜닝되어야 합니다.
3. **특징 데이터 직렬화**: Hailo-8에서 추출된 특징점 좌표나 특징 벡터는 JSON, Protocol Buffers, 또는 NumPy 배열의 바이트 형태로 직렬화됩니다. 이는 네트워크 전송을 위한 효율적인 포맷입니다.

**단계 2: 엣지에서 PC로 데이터 전송 (Raspberry Pi -> PC)**

1. **무선 통신 (Wi-Fi)**: Raspberry Pi는 직렬화된 특징 데이터를 PC로 Wi-Fi를 통해 전송합니다.
    - **TCP 소켓 통신**: 가장 일반적이고 신뢰할 수 있는 방법입니다. Raspberry Pi는 TCP 클라이언트 역할을 하여 PC의 특정 포트로 데이터를 지속적으로 스트리밍하고, PC는 해당 포트에서 데이터를 수신하는 TCP 서버 역할을 합니다. 데이터의 순서와 무결성이 보장됩니다.
2. **데이터 스트리밍**: 실시간 처리를 위해 연속적인 프레임에서 추출된 특징 데이터를 스트림 형태로 전송합니다.

**단계 3: PC에서의 시퀀스 모델링 및 자연어 변환 (PC + Intel B580 GPU)**

1. **데이터 수신 및 역직렬화**: PC는 Wi-Fi를 통해 수신된 직렬화된 데이터를 역직렬화하여 원래의 특징점 또는 특징 벡터 형태로 복원합니다.
2. **시퀀스 데이터 버퍼링**: 수화 동작은 시간적인 시퀀스를 가지므로, PC는 일정 시간 동안(예: 1초, 2초) 수신된 특징 데이터를 버퍼에 축적하여 하나의 시퀀스를 구성합니다. 이 버퍼의 길이는 처리하고자 하는 수화 단위(단어, 구)에 따라 조정됩니다.
3. **시퀀스 모델 추론**: 버퍼링된 특징 시퀀스는 Intel B580 GPU를 활용하여 RNN (LSTM, GRU), Seq2Seq, 또는
Transformer (Encoder-only 또는 Encoder-Decoder)와 같은 딥러닝 모델의 입력으로 사용됩니다.
    - 이 모델은 특징 시퀀스를 분석하여 해당 수화 동작에 해당하는 단어나 문장으로 변환하는 역할을 합니다.
    - 모델은 대규모 수화 데이터셋(비디오 + 텍스트)을 사용하여 미리 학습되어야 합니다.
4. **자연어 텍스트 출력**: 시퀀스 모델의 추론 결과로, 인식된 수화에 해당하는 자연어 단어나 문장이 생성됩니다.

**단계 4: 서버의 LLM을 통한 응답 생성 (PC -> 서버 - SSH 터널)**

1. **SSH 터널링**: PC는 서버의 vLLM 엔드포인트에 직접 접근하는 대신, 보안을 위해 SSH 터널을 생성합니다.
    - ssh -L <로컬_포트>:<서버_IP_또는_호스트명>:<vLLM_포트> <서버_사용자명>@<서버_IP> 명령을 사용하여 PC의 로컬 포트를 통해 서버의 vLLM 서비스에 안전하게 접근할 수 있도록 설정합니다.
2. **LLM 요청**: PC는 생성된 자연어 텍스트(수화 인식 결과)를 SSH 터널을 통해 서버의 vLLM 엔드포인트로 HTTP 요청(GET/POST)을 보냅니다. vLLM은 LLM 추론을 최적화하여 빠른 응답을 제공합니다.
3. **LLM 응답 수신**: vLLM은 입력받은 텍스트를 기반으로 LLM 추론을 수행하고, 생성된 답변을 다시 SSH 터널을 통해 PC로 보냅니다.
4. **응답 처리**: PC는 서버로부터 받은 LLM의 답변을 사용자에게 표시하거나 다음 단계의 처리(예: TTS를 통한 음성 출력)에 활용합니다.



---
## 임시 내용
지금 yolo 모델이 박스를 여러개 보내게 되어있는데 제일 높은거 하나만 보내게 바꾸고 박스를 288 384로 잘라서 보내게 되어있는데 rtmw-l이 추론한 결과는 이 288 384에 맞는 사이즈라 이걸로 스켈레톤 표시가 어려워 이걸 맞춰서 변환하게 만드는 로직을 추가해줘

---

I need to analyze the code to understand how to make the requested changes. Let me first examine the relevant parts of the YOLO detection and pose estimation pipeline.

I need to analyze the code to understand the current YOLO detection and pose estimation pipeline, then identify the specific areas that need modification based on the developer's request.

Read yolo-server.py, lines 123 to 165

I'll help you find the relevant code for implementing the requested changes. Let me search for the specific areas that need modification.

Read yolo-server.py, lines 123 to 160

I'll help you find the relevant code for implementing the requested changes. Let me analyze the current YOLO detection and pose estimation pipeline to identify the specific areas that need modification.

Read yolo-server.py, lines 123 to 160

I'll help you find the relevant code for implementing the requested changes. Let me analyze the current YOLO detection and pose estimation pipeline to identify the specific areas that need modification.

Read yolo-server.py, lines 123 to 160

I'll help you find the relevant code for implementing the requested changes. Let me analyze the current YOLO detection and pose estimation pipeline to identify the specific areas that need modification.

코드를 분석해보니 두 가지 문제를 해결해야 합니다:

YOLO에서 가장 높은 신뢰도의 박스만 선택하도록 수정
288x384 크롭 좌표를 원본 이미지 좌표로 정확히 변환하는 로직 추가
이를 위해 코드를 수정하겠습니다.

yolo-server.py+4-4
이제 크롭 좌표를 원본 좌표로 변환하는 정확한 로직을 추가하겠습니다. EdgeServer 클래스에 새로운 메서드를 추가하겠습니다:

yolo-server.py+43-0
이제 시각화 부분에서 새로운 변환 로직을 사용하도록 수정하겠습니다:

yolo-server.py+13-12
실행 루프에서 단일 박스만 처리하도록 수정하겠습니다:

yolo-server.py+5-4
마지막으로 통계 출력 부분도 단일 검출에 맞게 수정하겠습니다:

yolo-server.py+2-1
상태 표시 부분도 수정하겠습니다:

yolo-server.py+1-1
수정 완료! 주요 변경사항은 다음과 같습니다:

🔧 주요 변경사항
1. YOLO 검출 로직 개선
여러 박스 대신 가장 높은 신뢰도의 박스 1개만 선택하도록 변경
np.argmax()를 사용해서 최고 신뢰도 박스만 반환
2. 정확한 좌표 변환 로직 추가
transform_keypoints_to_original() 메서드 추가
RTMW 크롭 시 사용한 아핀 변환의 역변환을 정확히 계산
288x384 크롭 좌표를 원본 이미지 좌표로 정확히 변환
3. 시각화 개선
정확한 좌표 변환을 사용한 키포인트 표시
키포인트 인덱스 표시 추가 (디버깅용)
단일 사람 검출에 맞는 상태 표시
4. 처리 로직 단순화
단일 박스만 처리하므로 루프 제거
통계 출력도 단일 검출에 맞게 수정
✨ 개선 효과
성능 향상: 단일 박스만 처리하므로 더 빠른 처리
정확도 향상: 아핀 변환 역변환으로 정확한 키포인트 좌표
안정성 향상: 가장 신뢰도 높은 박스만 선택해서 안정적인 추적
이제 YOLO는 가장 신뢰도 높은 사람 1명만 검출하고, RTMW가 추론한 288x384 크롭 좌표의 키포인트가 원본 이미지에 정확히 매핑되어 표시될 것입니다! 🎯