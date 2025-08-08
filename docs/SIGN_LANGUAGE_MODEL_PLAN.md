# 🎯 수화 데이터 통합 분석 및 모델 개발 계획

## 📊 현재 데이터 구조 분석

### 1️⃣ Sign Language Dataset (LZF) 구조
```
📁 sign_language_dataset_lzf.h5
├── vocabulary/
│   ├── words: [3,303개 단어] - 빈도순 정렬
│   └── frequencies: 각 단어별 출현 빈도
├── segments/ 
│   ├── data_types: [149,874개] - 0=WORD, 1=SEN
│   ├── data_ids: WORD1234 또는 SEN1234
│   ├── real_ids: REAL01~16 (참여자 ID)
│   ├── views: 0=F, 1=U, 2=D, 3=L, 4=R (시점)
│   ├── start_frames: 시작 프레임 번호
│   ├── end_frames: 끝 프레임 번호  
│   ├── duration_frames: 지속 시간(프레임)
│   ├── vocab_ids: vocabulary ID 배열 (패딩)
│   └── vocab_lens: 실제 단어 개수
└── metadata/
    ├── total_segments: 149,874
    ├── vocabulary_size: 3,303
    └── fps: 30
```

**특징:**
- **morpheme 데이터 기반**: 수화 단어/문장의 의미론적 레이블
- **시간 정보**: start_frame, end_frame으로 정확한 시간 범위
- **참여자별**: REAL01~16 (16명의 다른 참여자)
- **다중 시점**: F(정면) 우선 선택됨
- **vocabulary 기반**: 3,303개 고유 단어의 ID 매핑

### 2️⃣ Pose Dataset 구조 (from output)
```
📁 batch_03_00_F_poses.h5
├── video_0001/
│   ├── frames: (140, 384, 288, 3) - RGB 프레임들
│   └── metadata: JSON 메타데이터
├── video_0002/
│   ├── frames: (141, 384, 288, 3)
│   └── metadata
├── ... (총 250개 비디오)
└── video_0250/
    ├── frames: (n_frames, 384, 288, 3)
    └── metadata
```

**특징:**
- **시각적 프레임 데이터**: 실제 RGB 이미지 시퀀스
- **고정 해상도**: 384x288 픽셀
- **가변 길이**: 각 비디오마다 다른 프레임 수 (78~189 프레임)
- **메타데이터**: 추가 정보 (JSON 형태)

## 🔗 데이터 연동 전략

### 핵심 연동 포인트

1. **시간 기반 매칭**
```python
# Sign Language의 프레임 범위
start_frame = segments['start_frames'][i]  # 예: 52
end_frame = segments['end_frames'][i]      # 예: 93

# Pose 데이터에서 해당 구간 추출
pose_frames = pose_data['video_xxxx']['frames'][start_frame:end_frame]
```

2. **참여자 ID 매칭**
```python
# Sign Language: REAL03 (real_ids = 3)
# Pose Data: batch_03_xx_F → 참여자 03
real_id = segments['real_ids'][i]  # 3
pose_file = f'batch_{real_id:02d}_xx_F_poses.h5'
```

3. **비디오 ID 매칭 (추정)**
```python
# Sign Language: data_id (WORD1234 → 1234)
# Pose Data: video_0001 ~ video_0250
data_id = segments['data_ids'][i]  # 1234
video_key = f'video_{data_id:04d}'  # video_1234
```

### 연동 데이터 구조 설계

```python
class UnifiedSignLanguageData:
    """통합 수화 데이터"""
    
    def __init__(self):
        self.segments = []  # 각 세그먼트 정보
        
    def load_segment(self, segment_idx):
        """특정 세그먼트의 통합 데이터 로드"""
        # 1. Sign Language 정보 추출
        segment_info = self.get_segment_info(segment_idx)
        
        # 2. 해당하는 Pose 데이터 찾기
        pose_data = self.get_pose_data(
            real_id=segment_info['real_id'],
            data_id=segment_info['data_id']
        )
        
        # 3. 시간 범위로 프레임 추출
        frames = pose_data['frames'][
            segment_info['start_frame']:segment_info['end_frame']
        ]
        
        return {
            'vocabulary_ids': segment_info['vocab_ids'],
            'vocabulary_words': [vocab[id] for id in segment_info['vocab_ids']],
            'frames': frames,  # (duration, 384, 288, 3)
            'duration': segment_info['duration'],
            'participant': f"REAL{segment_info['real_id']:02d}",
            'data_type': 'WORD' if segment_info['data_type'] == 0 else 'SEN'
        }
```

## 🏗️ 수화 모델 아키텍처 계획

### Phase 1: 데이터 통합 파이프라인 구축

#### 1.1 통합 데이터 로더 개발
```python
# 파일: unified_dataloader.py
class UnifiedSignLanguageDataset(Dataset):
    """Sign Language + Pose 통합 데이터셋"""
    
    def __init__(self, 
                 sign_lang_path: str,
                 pose_data_dir: str,
                 transform=None):
        # Sign Language 메타데이터 로드
        # Pose 데이터 인덱싱
        # 매칭 테이블 생성
        
    def __getitem__(self, idx):
        # 통합된 데이터 반환
        return {
            'frames': torch.tensor,      # (T, H, W, C)
            'vocab_ids': torch.tensor,   # (V,)
            'vocab_words': List[str],    # 실제 단어들
            'duration': int,
            'participant_id': int,
            'data_type': str
        }
```

#### 1.2 데이터 검증 및 품질 관리
- **매칭 검증**: Sign Language와 Pose 데이터 매칭률 확인
- **프레임 일관성**: 시간 범위와 실제 프레임 수 일치 확인  
- **데이터 품질**: 손실된 프레임, 노이즈 데이터 필터링

### Phase 2: 기본 수화 인식 모델

#### 2.1 Baseline 모델: CNN-LSTM
```python
class SignLanguageBaseline(nn.Module):
    """기본 수화 인식 모델"""
    
    def __init__(self, vocab_size=3303, hidden_dim=512):
        super().__init__()
        
        # CNN 백본 (프레임별 특징 추출)
        self.cnn_backbone = torchvision.models.resnet18(pretrained=True)
        self.cnn_backbone.fc = nn.Linear(512, hidden_dim)
        
        # LSTM (시퀀스 모델링)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, 
                           num_layers=2, batch_first=True, 
                           dropout=0.2, bidirectional=True)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, frames):
        # frames: (B, T, H, W, C)
        B, T = frames.shape[:2]
        
        # CNN으로 프레임별 특징 추출
        frames_flat = frames.view(B*T, *frames.shape[2:])
        features = self.cnn_backbone(frames_flat)  # (B*T, hidden_dim)
        features = features.view(B, T, -1)  # (B, T, hidden_dim)
        
        # LSTM으로 시퀀스 모델링
        lstm_out, _ = self.lstm(features)  # (B, T, hidden_dim*2)
        
        # 마지막 타임스텝으로 분류
        output = self.classifier(lstm_out[:, -1])  # (B, vocab_size)
        
        return output
```

#### 2.2 학습 전략
- **Task**: 단어/문장 분류 (3,303 클래스)
- **Loss**: CrossEntropy + Label Smoothing
- **Optimizer**: AdamW with Cosine LR Schedule
- **Data Augmentation**: 시간적 왜곡, 프레임 드롭아웃

### Phase 3: 고급 모델 - Transformer 기반

#### 3.1 Video Transformer 아키텍처
```python
class VideoTransformer(nn.Module):
    """비디오 트랜스포머 기반 수화 인식"""
    
    def __init__(self, vocab_size=3303):
        super().__init__()
        
        # Patch Embedding (Video ViT 방식)
        self.patch_embed = VideoPatchEmbed(
            img_size=288, patch_size=16, 
            in_chans=3, embed_dim=768
        )
        
        # Spatial-Temporal Attention
        self.st_blocks = nn.ModuleList([
            SpatialTemporalBlock(dim=768, num_heads=12)
            for _ in range(12)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, vocab_size)
        
    def forward(self, x):
        # x: (B, T, H, W, C)
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Transformer blocks
        for block in self.st_blocks:
            x = block(x)
        
        # Global average pooling + classification
        x = self.norm(x.mean(dim=1))
        return self.head(x)
```

#### 3.2 Multi-Modal Fusion
```python
class MultiModalSignModel(nn.Module):
    """텍스트(vocabulary) + 비디오 융합"""
    
    def __init__(self):
        super().__init__()
        
        # Video encoder
        self.video_encoder = VideoTransformer()
        
        # Text encoder (vocabulary embedding)
        self.text_encoder = nn.TransformerEncoder(...)
        
        # Cross-modal fusion
        self.fusion = CrossModalAttention()
        
        # Final classifier
        self.classifier = nn.Linear(fusion_dim, vocab_size)
```

### Phase 4: 실시간 수화 번역 시스템

#### 4.1 스트리밍 파이프라인
```python
class RealTimeSignTranslator:
    """실시간 수화 번역"""
    
    def __init__(self):
        self.model = trained_model
        self.buffer = FrameBuffer(max_frames=150)  # 5초 @ 30fps
        self.vocabulary = load_vocabulary()
        
    def process_frame(self, frame):
        """새 프레임 처리"""
        self.buffer.add_frame(frame)
        
        # 충분한 프레임이 쌓이면 추론
        if len(self.buffer) >= min_frames:
            prediction = self.model(self.buffer.get_sequence())
            words = self.decode_prediction(prediction)
            return words
```

#### 4.2 Edge 배포 최적화
- **모델 경량화**: Knowledge Distillation, Pruning
- **양자화**: INT8 quantization for faster inference
- **하드웨어 최적화**: Hailo-8 NPU 컴파일 (기존 hailotest.py 활용)

## 📈 개발 로드맵

### 단계 1: 데이터 통합 (2-3주)
- [ ] 통합 데이터 로더 개발
- [ ] 데이터 매칭 검증 도구
- [ ] 전처리 파이프라인 구축
- [ ] 데이터셋 분할 (Train/Val/Test)

### 단계 2: Baseline 모델 (2-3주)  
- [ ] CNN-LSTM 기본 모델 구현
- [ ] 학습 파이프라인 구축
- [ ] 평가 메트릭 정의
- [ ] 첫 번째 성능 벤치마크

### 단계 3: 고급 모델 (3-4주)
- [ ] Video Transformer 구현
- [ ] Multi-modal fusion 실험
- [ ] 하이퍼파라미터 튜닝
- [ ] 성능 비교 분석

### 단계 4: 실서비스 준비 (2-3주)
- [ ] 실시간 추론 파이프라인
- [ ] Edge 디바이스 최적화
- [ ] API 서버 통합
- [ ] 성능 모니터링

## 🎯 예상 성과

### 기술적 목표
- **정확도**: Top-1 85%+ (3,303 클래스 분류)
- **속도**: 실시간 처리 (<100ms latency)
- **효율성**: Edge device 배포 가능

### 활용 방안
1. **수화 교육**: 실시간 수화 학습 피드백
2. **의사소통 도구**: 청각 장애인-일반인 소통
3. **미디어 접근성**: 자동 수화 번역 서비스
4. **연구 플랫폼**: 수화 언어학 연구 도구

---

## 🚀 시작하기

### 필수 의존성
```bash
pip install torch torchvision torchaudio
pip install h5py numpy pandas opencv-python
pip install transformers timm
pip install mmpose  # pose estimation
pip install ultralytics  # YOLO
```

### 첫 번째 실행
```bash
# 1. 데이터 통합 확인
python unified_dataloader.py --validate

# 2. 기본 모델 학습
python train_baseline.py --config configs/baseline.yaml

# 3. 평가
python evaluate.py --model checkpoints/best_model.pth
```

이 계획은 현실적이고 단계별로 진행 가능하며, 기존의 pose-server.py와 yolo-server.py 인프라를 최대한 활용합니다.
