# 🚀 수화 인식 AI 모델 - 실행 가이드

## 📁 프로젝트 구조

```
mosu/
├── 📊 데이터 처리
│   ├── mosuModel/data/
│   │   ├── sign_language_dataset_lzf.h5      # 수화 메타데이터 (LZF 압축)
│   │   ├── poses/batch_03_*_F_poses.h5       # 포즈/프레임 데이터
│   │   ├── unified_dataloader.py             # 통합 데이터 로더
│   │   ├── pytorch_dataloader.py             # 기본 PyTorch 로더
│   │   └── complete_data_pipeline.py         # 데이터 파이프라인
│   │
├── 🧠 AI 모델
│   ├── mosuModel/
│   │   ├── sign_language_models.py           # 모델 아키텍처들
│   │   ├── train_sign_language.py            # 학습 스크립트
│   │   ├── evaluate_sign_language.py         # 평가 스크립트
│   │   └── sign_translation_server.py        # 실시간 번역 서버
│   │
├── 🎥 실시간 처리
│   ├── pose-server.py                        # 포즈 추정 서버
│   ├── yolo-server.py                        # 사람 검출 서버
│   └── cmd-pose, cmd-yolo                    # 실행 스크립트들
│   │
└── 📚 문서
    ├── docs/SIGN_LANGUAGE_MODEL_PLAN.md      # 전체 계획서
    └── docs/PROJECT_PLAN.md                  # 프로젝트 계획
```

## 🎯 단계별 실행 가이드

### Phase 1: 환경 설정

```bash
# 1. Python 환경 설정
cd /home/jy/gitwork/mosu
python -m venv .venv
source .venv/bin/activate

# 2. 의존성 설치
pip install torch torchvision torchaudio
pip install h5py numpy pandas opencv-python
pip install flask requests tqdm
pip install matplotlib seaborn scikit-learn
pip install tensorboard
pip install mmpose ultralytics  # 포즈 추정용

# 3. 데이터 확인
python mosuModel/data/unified_dataloader.py
```

### Phase 2: 데이터 검증 및 통합

```bash
# 1. 데이터 구조 확인
cd mosuModel/data
python -c "
from unified_dataloader import create_unified_dataloader
loader, dataset = create_unified_dataloader(
    'sign_language_dataset_lzf.h5',
    'poses/',
    batch_size=4,
    validate_matching=True
)
print('Dataset size:', len(dataset))
print('Stats:', dataset.get_stats())
"

# 2. 첫 번째 배치 테스트
python unified_dataloader.py
```

### Phase 3: 모델 학습

```bash
# 1. Baseline CNN-LSTM 모델 학습
python mosuModel/train_sign_language.py \
  --sign-lang-path mosuModel/data/sign_language_dataset_lzf.h5 \
  --pose-data-dir mosuModel/data/poses/ \
  --model-type baseline \
  --batch-size 8 \
  --num-epochs 30 \
  --learning-rate 1e-3 \
  --max-frames 100 \
  --save-dir checkpoints/baseline \
  --log-dir logs/baseline

# 2. Transformer 모델 학습
python mosuModel/train_sign_language.py \
  --sign-lang-path mosuModel/data/sign_language_dataset_lzf.h5 \
  --pose-data-dir mosuModel/data/poses/ \
  --model-type transformer \
  --batch-size 6 \
  --num-epochs 50 \
  --learning-rate 5e-4 \
  --max-frames 100 \
  --save-dir checkpoints/transformer \
  --log-dir logs/transformer

# 3. Multi-modal 모델 학습
python mosuModel/train_sign_language.py \
  --sign-lang-path mosuModel/data/sign_language_dataset_lzf.h5 \
  --pose-data-dir mosuModel/data/poses/ \
  --model-type multimodal \
  --batch-size 4 \
  --num-epochs 40 \
  --learning-rate 3e-4 \
  --max-frames 100 \
  --save-dir checkpoints/multimodal \
  --log-dir logs/multimodal
```

### Phase 4: 모델 평가

```bash
# 1. 학습된 모델 평가
python mosuModel/evaluate_sign_language.py \
  --checkpoint checkpoints/baseline/best_model.pth \
  --sign-lang-path mosuModel/data/sign_language_dataset_lzf.h5 \
  --pose-data-dir mosuModel/data/poses/ \
  --model-type baseline \
  --batch-size 16 \
  --results-dir evaluation_results/baseline

# 2. Transformer 모델 평가  
python mosuModel/evaluate_sign_language.py \
  --checkpoint checkpoints/transformer/best_model.pth \
  --sign-lang-path mosuModel/data/sign_language_dataset_lzf.h5 \
  --pose-data-dir mosuModel/data/poses/ \
  --model-type transformer \
  --batch-size 16 \
  --results-dir evaluation_results/transformer

# 3. 결과 확인
cat evaluation_results/baseline/metrics.json
```

### Phase 5: 실시간 번역 서버 실행

```bash
# 1. Vocabulary JSON 생성
python -c "
from mosuModel.data.unified_dataloader import create_unified_dataloader
_, dataset = create_unified_dataloader('mosuModel/data/sign_language_dataset_lzf.h5', 'mosuModel/data/poses/')
import json
with open('vocabulary.json', 'w', encoding='utf-8') as f:
    json.dump(dataset.get_vocabulary(), f, ensure_ascii=False, indent=2)
print('Vocabulary saved to vocabulary.json')
"

# 2. 실시간 번역 서버 실행
python mosuModel/sign_translation_server.py \
  --checkpoint checkpoints/baseline/best_model.pth \
  --model-type baseline \
  --vocabulary vocabulary.json \
  --host 0.0.0.0 \
  --port 5001 \
  --max-frames 100 \
  --min-frames 30 \
  --confidence-threshold 0.1
```

### Phase 6: 전체 시스템 통합

```bash
# Terminal 1: 포즈 추정 서버
python pose-server.py \
  --config configs/rtmw-l_8xb320-270e_cocktail14-384x288.py \
  --checkpoint models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-f840f204_20231122.pth \
  --port 5000

# Terminal 2: 수화 번역 서버  
python mosuModel/sign_translation_server.py \
  --checkpoint checkpoints/baseline/best_model.pth \
  --model-type baseline \
  --vocabulary vocabulary.json \
  --port 5001

# Terminal 3: 엣지 클라이언트 (YOLO + 통합)
python yolo-server.py \
  --pose-server http://localhost:5000 \
  --translation-server http://localhost:5001 \
  --camera 0
```

## 📊 성능 모니터링

### TensorBoard 로그 확인
```bash
# 학습 진행 상황 모니터링
tensorboard --logdir logs --port 6006
# http://localhost:6006 에서 확인
```

### API 테스트
```bash
# 1. 헬스 체크
curl http://localhost:5001/health

# 2. 통계 확인
curl http://localhost:5001/stats

# 3. Vocabulary 확인
curl http://localhost:5001/vocabulary

# 4. 이미지 번역 테스트
curl -X POST -F "image=@test_frame.jpg" http://localhost:5001/translate
```

## 🎯 예상 결과

### 학습 성능
- **Baseline (CNN-LSTM)**: 70-80% 정확도
- **Transformer**: 75-85% 정확도  
- **Multi-modal**: 80-90% 정확도

### 실시간 성능
- **추론 속도**: 10-50ms (GPU 기준)
- **처리량**: 20-100 FPS
- **메모리 사용**: 2-8GB (모델에 따라)

### 시스템 지연시간
- **전체 파이프라인**: 100-300ms
  - YOLO 검출: 20-50ms
  - 포즈 추정: 30-80ms  
  - 수화 인식: 50-150ms

## 🔧 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   --batch-size 2
   # 또는 CPU 사용
   --device cpu
   ```

2. **데이터 로딩 속도 느림**
   ```bash
   # Worker 수 줄이기
   --num-workers 2
   ```

3. **h5py 설치 오류**
   ```bash
   pip install h5py==3.8.0
   ```

4. **포즈 추정 모델 다운로드**
   ```bash
   mkdir -p models
   wget https://download.openmmlab.com/mmpose/v1/whole_body_2d_keypoint/rtmpose/cocktail14/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-f840f204_20231122.pth -P models/
   ```

## 🚀 다음 단계

1. **모델 최적화**
   - Knowledge Distillation으로 경량화
   - INT8 Quantization 적용
   - ONNX/TensorRT 변환

2. **데이터 확장**
   - 더 많은 참여자 데이터 추가
   - 다양한 시점 데이터 활용
   - Data Augmentation 강화

3. **실서비스 배포**
   - Docker 컨테이너화
   - Kubernetes 배포
   - Load Balancer 구성

4. **사용자 인터페이스**
   - 웹 인터페이스 개발
   - 모바일 앱 연동
   - 실시간 피드백 시스템

---

이 가이드를 따라하면 완전한 수화 인식 AI 시스템을 구축할 수 있습니다! 🎉
