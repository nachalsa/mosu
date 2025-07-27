# **Hailo-8 지원/미지원 연산 정리 (테스트 결과 기반 )**
*https://github.com/SeSac01/-/edit/main/docs/hardware/testcode/hailotest.py*

이 문서는 다음 환경에서 수행된 PyTorch 모델 테스트 스크립트의 실행 결과를 바탕으로 Hailo-8 AI 가속기에서 지원되거나 문제없이 컴파일된 연산들과, 컴파일에 실패하거나 특정 조건에서 문제가 발생한 연산들을 종합하여 정리합니다.

**테스트 환경:**
*   **Linux Kernel 버전:** 6.15.5-061505-generic
*   **Hailo SDK 실행 환경:** `hailo8_ai_sw_suite_2025-07_docker.zip` 패키지에 포함된 `hailo_ai_sw_suite_docker_run.sh` 스크립트를 통해 생성된 **Docker 컨테이너 내부**에서 실행되었습니다.
*   **PyTorch 버전:** 2.7.1+cu126
*   **ONNX Library 버전:** 1.16.0 (ONNX opset version 11로 export 됨)
*   **HailoRT (Runtime) 버전:** v4.22.0
*   **Hailo Dataflow Compiler 버전:** v3.32.0

---

## ✅ 완전 지원 연산

아래 연산들은 **테스트 코드에서 ONNX 변환 및 Hailo 컴파일이 성공적으로 완료**되었습니다. 이는 Hailo-8 AI 가속기에서 해당 연산들이 기본적인 형태로 잘 지원됨을 의미합니다.

### **I. Convolution 계열 (`torch.nn` Modules)**

*   **`nn.Conv2d`** (Standard 2D Convolution)
    *   일반적인 `kernel_size` (예: 1x1, 3x3, 5x5), `padding`, `stride` 조합.
    *   **Depthwise Convolution** (`groups=in_channels` 사용).
    *   **Grouped Convolution** (`groups > 1` and `groups < in_channels` 사용).
    *   **Dilated Convolution** (`dilation > 1` 사용).
*   **`nn.ConvTranspose2d`** (Transpose Convolution / Deconvolution)

### **II. Pooling 연산 (`torch.nn` Modules & `torch.nn.functional` Functions)**

*   **`nn.MaxPool2d`** (Max Pooling)
    *   다양한 `kernel_size`, `stride`, `padding` 조합.
*   **`nn.AvgPool2d`** (Average Pooling)
    *   다양한 `kernel_size`, `stride`, `padding` 조합.
*   **`nn.AdaptiveMaxPool2d`** (Adaptive Max Pooling, 특정 출력 크기 지정).
*   **`nn.AdaptiveAvgPool2d`** (Adaptive Average Pooling, 특히 `(1,1)` Global Average Pooling).
*   **`F.max_pool2d`** (`torch.nn.functional.max_pool2d`).

### **III. Activation 함수 (`torch.nn` Modules & `torch.nn.functional` / `torch` Functions)**

*   **`F.relu`** (ReLU).
*   **`F.leaky_relu`** (Leaky ReLU).
*   **`torch.sigmoid`** (Sigmoid).
*   **`torch.tanh`** (Tanh).
*   **`nn.Hardswish`** (HardSwish).
*   **`nn.Hardsigmoid`** (HardSigmoid).
*   **`F.gelu`** (GELU).
*   **`F.elu`** (ELU).
*   **`nn.PReLU`** (PReLU).
*   **Swish/SiLU** (요소별 연산 (`x * torch.sigmoid(x)`) 조합으로 구현되어 지원).

### **IV. Normalization (`torch.nn` Modules)**

*   **`nn.BatchNorm2d`** (Batch Normalization).
*   **`nn.LayerNorm`** (Layer Normalization).
*   **`nn.InstanceNorm2d`** (Instance Normalization).

### **V. Linear 및 Flatten 연산 (`torch.nn` Modules)**

*   **`nn.Linear`** (Fully Connected / Dense Layer)
    *   테스트된 모델 크기 내에서 성공적으로 컴파일됨.
*   **`nn.Flatten`** (Flatten Operation).

### **VI. Element-wise 및 기타 Tensor 연산 (`Python` Operators & `torch` Functions)**

*   **`+`** (덧셈, `torch.add`).
*   **`-`** (뺄셈, `torch.sub`).
*   **`*`** (곱셈, `torch.mul`).
*   **`/`** (나눗셈, `torch.div`).
*   **`torch.cat`** (Concatenation).
*   **`torch.squeeze`** (크기가 1인 차원 제거, 예를 들어 `x.squeeze(-1).squeeze(-1)`).
*   **`torch.permute`** (차원 순서 변경).
*   **`nn.ZeroPad2d`** (명시적인 Zero Padding).
*   **`torch.clamp`** (값 범위 제한).
*   **`torch.reshape`** (텐서 모양 변경, 단 `torch.mean` 등 특정 연산과의 조합 시 문제가 발생할 수 있음).

### **VII. Resize / Interpolation (`torch.nn.functional` Functions)**

*   **`F.interpolate`** (Bilinear Mode).
*   **`F.interpolate`** (Nearest Mode).

### **VIII. 기타 지원 연산 (패턴 포함)**

*   **`nn.Dropout2d`** (Dropout, 추론 시에는 일반적으로 No-Op으로 처리).
*   **Residual Block 패턴** (덧셈을 통한 Skip Connection).
*   **Dense Connection 패턴** (Concatenation을 통한 Skip Connection).
*   **Simple Attention 패턴** (요소별 곱셈을 포함).

---

## ❌ 미지원 또는 문제 발생 연산

아래 연산들은 **테스트 코드에서 컴파일에 실패**했습니다.

*   **`nn.Conv1d`** (1D Convolution)
    *   **문제:** 컴파일 단계에서 **타임아웃** 발생. 이는 Hailo 컴파일러가 해당 연산을 처리하는 데 과도한 시간이 소요되거나, 효율적인 하드웨어 매핑에 어려움이 있음을 시사합니다.
    *   **대안:** 2D Convolution (`nn.Conv2d`에 특정 `kernel_size`, `padding` 조합 사용)으로 우회하는 것을 고려할 수 있습니다.
*   **`torch.mean(dim=...)`** (특정 차원에서의 Mean Reduction)
    *   **문제:** ONNX 변환은 성공했으나, Hailo **파싱(Parse) 단계에서 `Invalid kernel shape` 오류** 발생.
    *   **원인 분석:** `torch.mean` 연산 자체는 지원되는 것으로 보이지만, 이 연산이 만들어내는 중간 텐서의 특정 내부 표현(예: `(B, C, L)` 형태)이 뒤이어 오는 `torch.reshape` 및 `nn.Linear` 레이어와의 조합에서 Hailo 컴파일러의 내부 `nn.Linear` 최적화 로직(이를 컨볼루션 레이어로 변환 시도)과 충돌하기 때문입니다.
    *   **대안:** `nn.AdaptiveAvgPool2d((1, 1))` 및 `torch.squeeze`를 사용하여 글로벌 평균 풀링을 수행한 후 `nn.Linear`에 연결하는 방식으로 우회하는 것을 권장합니다. 이는 Hailo 컴파일러가 가장 안정적으로 처리하는 표준적인 패턴입니다.
*   **`torch.Tensor.view`** (일반적인 `view` 메서드)
    *   **문제:** 초기 `basic_conv` 모델에서 **ONNX 변환(Export) 단계**에서 실패한 이력이 있습니다.
    *   **원인 분석:** `torch.onnx.export`가 `view` 연산을 ONNX 그래프로 내보내는 과정에서 특정 조건(예: 텐서의 메모리 연속성)이나 버전 호환성 문제로 불안정할 수 있습니다.
    *   **대안:** 기능적으로 유사한 `torch.squeeze` 또는 `torch.reshape`를 사용하는 것이 ONNX 변환 및 Hailo 컴파일 과정에서 더 안정적입니다.

---

## ℹ️ 이 테스트에서 다루지 않았으나 일반적인 Hailo-8 특성 (참고용)

다음은 이 테스트 코드에 직접적으로 포함되지 않았지만, Hailo-8과 같은 신경망 처리 장치(NPU)에서 일반적으로 고려해야 할 지원 특성입니다.

### **제한적/조건부 지원 (일반적인 NPU 특성)**

*   **`nn.Linear` (매우 큰 크기):** 테스트에서는 성공했으나, 매우 큰 크기의 `nn.Linear` 레이어는 NPU 메모리 제약이나 효율성 문제로 인해 CPU로 폴백되거나 성능 저하가 발생할 수 있습니다. 1x1 Convolution + Global Average Pooling으로 변환을 고려할 수 있습니다.
*   **Custom Activation:** 사용자 정의 활성화 함수는 일반적으로 직접적으로 미지원이며, Hailo SDK가 제공하는 확장 메커니즘을 통해 구현해야 할 수 있습니다.

### **일반적으로 미지원 (일반적인 NPU 특성)**

*   **복잡한 수학 연산:** 고유값 분해 (Eigenvalue Decomposition), 고속 푸리에 변환 (FFT/IFFT), 단시간 푸리에 변환 (STFT) 등은 NPU에서 직접 지원되지 않는 경우가 많습니다.
*   **고급 제어 흐름 연산:**
    *   **Dynamic Shape Operations:** 입력 텐서의 모양이 추론 시점에 동적으로 변하는 연산은 일반적으로 고정된(Static) 입력 크기를 선호합니다.
    *   **Control Flow:** 조건문 (`if`), 반복문 (`for`, `while`)과 같은 프로그램 로직은 하드웨어 그래프로 변환하기 어렵습니다.
    *   **Custom Operations:** 사용자 정의 연산은 Hailo SDK가 제공하는 확장 메커니즘을 사용하지 않는 한 미지원입니다.
    *   **Variable-length Sequences:** 가변 길이 시퀀스 처리는 패딩 등의 전처리 없이는 NPU에서 직접 처리하기 어렵습니다.

---

## 🔄 대안 처리 방식 및 최적화 권장사항

NPU에서 모델을 최대한 효율적으로 실행하기 위해, 미지원 또는 비효율적인 연산을 지원되는 연산들로 **재구현(Re-implementation)하거나 모델 구조를 재구성(Reformulation)하는 것이 가장 권장**됩니다.

*   **모델 구조 재구성 및 NPU 친화적 디자인 (가장 권장되는 접근)**:
    *   **RNN, Transformer 등 시퀀스 모델 전환:** 시계열 또는 시퀀스 처리 시 RNN, LSTM, GRU, 복잡한 Attention 메커니즘 등 NPU에서 비효율적이거나 미지원인 순환/복잡한 구조 대신, TCN(Temporal Convolution Network)이나 Convolution 및 행렬 곱셈 등 Hailo에서 잘 지원되는 연산들로 구성된 아키텍처를 사용하는 것을 적극적으로 고려해야 합니다. 즉, 기존 RNN/Transformer와 *동일한 방식의 연산*을 직접 사용하기보다는, Hailo가 지원하는 연산들의 조합으로 *동일한 기능*을 수행하는 새로운 방식을 구현하는 것이 NPU 활용에 유리합니다.
    *   **Fully Connected 레이어 최적화:** 매우 큰 FC 레이어는 `nn.Conv2d(..., kernel_size=1, ...)` (1x1 convolution)와 `nn.AdaptiveAvgPool2d((1,1))` (Global Average Pooling) 조합으로 변환하여 NPU 친화적으로 만듭니다.
    *   **동적 크기 → 고정 크기:** 가능한 경우 모델 입력 및 중간 텐서의 크기를 고정된(Static) 크기로 변경하여 컴파일러의 최적화를 용이하게 합니다.

*   **하이브리드 실행 (Fallback 옵션)**:
    *   Hailo 컴파일러는 모델 내에 미지원 연산이 있을 경우, 해당 부분을 CPU에서 실행하고 지원되는 연산은 Hailo-8에서 하드웨어 가속하도록 **자동으로 모델을 분할하는 기능**을 제공합니다.
    *   이는 모델 전체를 NPU에 올리기 어렵거나, 미지원 부분이 매우 작고 성능에 큰 영향을 미치지 않을 때 유용합니다. 그러나 CPU 오버헤드로 인해 전체적인 성능 저하가 발생할 수 있으므로, **최적의 성능을 위해서는 위에서 언급된 NPU 친화적인 모델 재구현이 우선적으로 고려**되어야 합니다.

*이 목록이 Hailo-8 기반의 딥러닝 모델 개발에 참고 자료가 되기를 바랍니다.*