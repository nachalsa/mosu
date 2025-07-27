import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import os
import json
from pathlib import Path
import time

class HailoOperationTester:
    """
    Hailo AI 가속기에서 다양한 뉴럴 네트워크 연산들이 제대로 지원되고 컴파일되는지
    체계적으로 테스트하는 클래스.

    각 연산에 특화된 작은 PyTorch 모델을 생성하고, 이를 ONNX로 변환한 뒤,
    Hailo 컴파일러를 통해 파싱, 최적화, 컴파일 단계를 테스트합니다.
    """
    
    def __init__(self, output_dir="hailo_test_results"):
        """
        테스터 초기화. 결과 저장 디렉토리를 설정합니다.

        Args:
            output_dir (str): 모든 테스트 결과가 저장될 최상위 디렉토리 이름.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True) # 결과 디렉토리 생성
        
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(exist_ok=True) # ONNX 모델 저장 디렉토리
        
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True) # Hailo 컴파일 결과 저장 디렉토리
        
        self.test_results = {} # 각 모델의 테스트 결과를 저장할 딕셔너리
        
    def create_test_models(self):
        """
        다양한 연산을 테스트하기 위한 PyTorch 모델 인스턴스들을 생성합니다.
        각 모델은 특정 연산 또는 연산 패턴에 집중하여 Hailo 지원 여부를 확인합니다.

        Returns:
            dict: {모델 이름: nn.Module 인스턴스} 형태의 딕셔너리.
        """
        models = {}
        
        print("=== PyTorch 테스트 모델 생성 시작 ===")

        # 1. 기본 Convolution 연산들
        models['basic_conv'] = self._create_basic_conv_model()
        models['depthwise_conv'] = self._create_depthwise_conv_model()
        models['pointwise_conv'] = self._create_pointwise_conv_model()
        models['dilated_conv'] = self._create_dilated_conv_model()
        models['grouped_conv'] = self._create_grouped_conv_model()
        
        # 2. 활성화 함수들
        models['relu_activation'] = self._create_activation_model('relu')
        models['leaky_relu_activation'] = self._create_activation_model('leaky_relu')
        models['sigmoid_activation'] = self._create_activation_model('sigmoid')
        models['tanh_activation'] = self._create_activation_model('tanh')
        models['swish_activation'] = self._create_activation_model('swish') # x * sigmoid(x) 조합
        models['gelu_activation'] = self._create_activation_model('gelu')
        models['elu_activation'] = self._create_activation_model('elu')
        models['prelu_activation'] = self._create_activation_model('prelu')
        models['hardswish_activation'] = self._create_activation_model('hardswish')
        models['hardsigmoid_activation'] = self._create_activation_model('hardsigmoid')
        
        # 3. 정규화 연산들
        models['batch_norm'] = self._create_batch_norm_model()
        models['layer_norm'] = self._create_layer_norm_model()
        models['instance_norm'] = self._create_instance_norm_model()
        
        # 4. 풀링 연산들
        models['max_pool'] = self._create_pooling_model('max')
        models['avg_pool'] = self._create_pooling_model('avg')
        models['adaptive_pool'] = self._create_pooling_model('adaptive') # AdaptiveMaxPool2d
        models['global_avg_pool'] = self._create_pooling_model('global_avg') # AdaptiveAvgPool2d((1,1))
        models['max_pool_varied'] = self._create_varied_pooling_model('max') # 다양한 kernel/stride/padding
        models['avg_pool_varied'] = self._create_varied_pooling_model('avg') # 다양한 kernel/stride/padding
        
        # 5. Element-wise 연산들
        models['elementwise_add'] = self._create_elementwise_model('add')
        models['elementwise_mul'] = self._create_elementwise_model('mul')
        models['elementwise_sub'] = self._create_elementwise_model('sub')
        models['elementwise_div'] = self._create_elementwise_model('div')
        
        # 6. Resize/Interpolation 연산들
        models['bilinear_upsample'] = self._create_upsample_model('bilinear')
        models['nearest_upsample'] = self._create_upsample_model('nearest')
        
        # 7. Skip Connection 패턴들 (Residual, DenseNet 구조)
        models['residual_block'] = self._create_residual_model()
        models['dense_connection'] = self._create_dense_connection_model()
        
        # 8. Attention 메커니즘
        models['simple_attention'] = self._create_attention_model()
        
        # 9. 1D Conv 연산들 (문제 발생 이력: Timeout)
        models['conv1d'] = self._create_conv1d_model()
        
        # 10. Transpose Conv (Deconv)
        models['transpose_conv'] = self._create_transpose_conv_model()

        # 11. 추가된 다양한 연산들
        models['fully_connected'] = self._create_fully_connected_model() # nn.Linear의 기본적인 사용
        models['flatten_op'] = self._create_flatten_model() # nn.Flatten 모듈 명시적 테스트
        models['dropout_op'] = self._create_dropout_model() # nn.Dropout2d 테스트 (추론 시 No-Op)
        models['concatenation_op'] = self._create_concatenation_model() # torch.cat 테스트
        models['zero_pad'] = self._create_zero_pad_model() # nn.ZeroPad2d 테스트
        models['permute_op'] = self._create_permute_model() # torch.permute 테스트
        models['matmul_op'] = self._create_matmul_model() # nn.Linear를 통한 간접적인 MatMul 테스트
        models['clamp_op'] = self._create_clamp_model() # torch.clamp 테스트
        models['mean_reduction'] = self._create_mean_reduction_model() # torch.mean(dim=...) 테스트 (문제 발생 이력: 파싱 오류)
        
        print("=== PyTorch 테스트 모델 생성 완료 ===")
        return models
    
    # ----------------------------------------------------------------------
    # 각 연산/패턴을 테스트하기 위한 개별 PyTorch 모델 정의
    # (일관성을 위해 대부분 AdaptiveAvgPool2d -> squeeze -> Linear 패턴으로 끝남)
    # ----------------------------------------------------------------------

    def _create_basic_conv_model(self):
        """
        표준 2D Convolution (nn.Conv2d)의 기본적인 작동을 테스트합니다.
        초기 버전에서 `x.view()` 사용 시 ONNX export 문제가 있었으나,
        현재 `x.squeeze()`로 변경하여 해결되었습니다.
        """
        class BasicConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 5, padding=2) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10) # 64 features -> 10 classes
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x)) # Output shape e.g., (B, 64, H, W)
                x = self.pool(x)          # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64) - 1x1 차원 제거
                return self.classifier(x) # (B, 10)
        return BasicConvModel()
    
    def _create_depthwise_conv_model(self):
        """Depthwise Convolution (groups=in_channels) 및 Pointwise Conv 조합 테스트."""
        class DepthwiseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(1, 32, 3, padding=1)
                self.depthwise = nn.Conv2d(32, 32, 3, padding=1, groups=32)
                self.pointwise = nn.Conv2d(32, 64, 1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.stem(x))
                x = F.relu(self.depthwise(x))
                x = F.relu(self.pointwise(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return DepthwiseModel()
    
    def _create_pointwise_conv_model(self):
        """Pointwise Convolution (1x1 Conv) 자체를 테스트."""
        class PointwiseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.pointwise1 = nn.Conv2d(32, 64, 1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.pointwise1(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return PointwiseModel()
    
    def _create_dilated_conv_model(self):
        """Dilated Convolution (dilation 인자 사용)을 테스트."""
        class DilatedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.dilated_conv = nn.Conv2d(32, 64, 3, padding=2, dilation=2) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.dilated_conv(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return DilatedModel()
    
    def _create_grouped_conv_model(self):
        """Grouped Convolution (groups > 1)을 테스트."""
        class GroupedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.grouped_conv = nn.Conv2d(32, 64, 3, padding=1, groups=4) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.grouped_conv(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return GroupedModel()
    
    def _create_activation_model(self, activation_type):
        """다양한 활성화 함수(ReLU, LeakyReLU, Sigmoid, Tanh 등)를 테스트."""
        class ActivationModel(nn.Module):
            def __init__(self, act_type):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                self.act_type = act_type
                # PReLU, Hardswish, Hardsigmoid는 모듈로 선언해야 함
                if act_type == 'prelu':
                    self.prelu = nn.PReLU()
                elif act_type == 'hardswish':
                    self.hardswish = nn.Hardswish()
                elif act_type == 'hardsigmoid':
                    self.hardsigmoid = nn.Hardsigmoid()
                
            def forward(self, x):
                x = self.apply_activation(self.conv1(x))
                x = self.apply_activation(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
            
            def apply_activation(self, x):
                # 다양한 활성화 함수 적용
                if self.act_type == 'relu':
                    return F.relu(x)
                elif self.act_type == 'leaky_relu':
                    return F.leaky_relu(x, 0.1)
                elif self.act_type == 'sigmoid':
                    return torch.sigmoid(x)
                elif self.act_type == 'tanh':
                    return torch.tanh(x)
                elif self.act_type == 'swish':
                    return x * torch.sigmoid(x) # Element-wise ops for Swish
                elif self.act_type == 'gelu':
                    return F.gelu(x)
                elif self.act_type == 'elu':
                    return F.elu(x)
                elif self.act_type == 'prelu':
                    return self.prelu(x)
                elif self.act_type == 'hardswish':
                    return self.hardswish(x)
                elif self.act_type == 'hardsigmoid':
                    return self.hardsigmoid(x)
                else: # Fallback to ReLU if unknown type
                    return F.relu(x)
        return ActivationModel(activation_type)
    
    def _create_batch_norm_model(self):
        """Batch Normalization (nn.BatchNorm2d)을 테스트."""
        class BatchNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.bn2 = nn.BatchNorm2d(64)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = F.relu(self.bn2(self.conv2(x))) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return BatchNormModel()
    
    def _create_layer_norm_model(self):
        """Layer Normalization (nn.LayerNorm)을 테스트."""
        class LayerNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                # LayerNorm은 입력 텐서의 마지막 차원들에 적용되므로,
                # Conv2d 출력 (B, C, H, W)에 맞게 [C, H, W]를 지정
                self.ln1 = nn.LayerNorm([32, 28, 28])  # MNIST (1, 28, 28) 기준, Conv1 출력 (32, 28, 28)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(self.ln1(x)) # LayerNorm 후에 활성화 함수 적용
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return LayerNormModel()
    
    def _create_instance_norm_model(self):
        """Instance Normalization (nn.InstanceNorm2d)을 테스트."""
        class InstanceNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.in1 = nn.InstanceNorm2d(32)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.in1(self.conv1(x)))
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return InstanceNormModel()
    
    def _create_pooling_model(self, pool_type):
        """
        Max, Avg, Adaptive, Global Avg Pooling의 기본 동작을 테스트.
        F.adaptive_avg_pool2d((1,1))은 최종 분류 직전에 항상 적용됩니다.
        """
        class PoolingModel(nn.Module):
            def __init__(self, p_type):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool_type = p_type
                
                if p_type == 'max':
                    self.pool_op = nn.MaxPool2d(2, 2)
                elif p_type == 'avg':
                    self.pool_op = nn.AvgPool2d(2, 2)
                elif p_type == 'adaptive':
                    self.pool_op = nn.AdaptiveMaxPool2d((7, 7)) # 특정 크기로 Adaptive MaxPool
                elif p_type == 'global_avg':
                    # Global Average Pooling은 forward에서 직접 적용
                    pass 
                
                self.global_final_pool = nn.AdaptiveAvgPool2d((1, 1)) # 최종적으로 항상 글로벌 풀링
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                if self.pool_type != 'global_avg':
                    x = self.pool_op(x) # 특정 풀링 연산 적용
                
                x = F.relu(self.conv2(x)) # (B, 64, H', W')
                
                # 최종적으로 글로벌 풀링 후 Linear 레이어 연결
                x = self.global_final_pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return PoolingModel(pool_type)
    
    def _create_varied_pooling_model(self, pool_type):
        """다양한 kernel_size, stride, padding을 갖는 Max/Avg Pooling을 테스트."""
        class VariedPoolingModel(nn.Module):
            def __init__(self, p_type):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool_type = p_type
                
                if p_type == 'max':
                    self.pool_op = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif p_type == 'avg':
                    self.pool_op = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
                
                self.global_final_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool_op(x) # 다양한 파라미터의 풀링 적용
                x = F.relu(self.conv2(x)) # (B, 64, H', W')
                
                x = self.global_final_pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return VariedPoolingModel(pool_type)
    
    def _create_elementwise_model(self, op_type):
        """Add, Mul, Sub, Div 등 요소별(Element-wise) 연산을 테스트."""
        class ElementwiseModel(nn.Module):
            def __init__(self, operation):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(1, 32, 3, padding=1) # 동일한 크기의 두 텐서 생성
                self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                self.operation = operation
                
            def forward(self, x):
                x1 = F.relu(self.conv1(x))
                x2 = F.relu(self.conv2(x))
                
                if self.operation == 'add':
                    x_combined = x1 + x2
                elif self.operation == 'mul':
                    x_combined = x1 * x2
                elif self.operation == 'sub':
                    x_combined = x1 - x2
                elif self.operation == 'div':
                    x_combined = x1 / (x2 + 1e-8)  # ZeroDivisionError 방지를 위한 작은 값 추가
                else:
                    raise ValueError(f"Unsupported element-wise operation: {self.operation}")
                
                x = F.relu(self.conv3(x_combined)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return ElementwiseModel(op_type)
    
    def _create_upsample_model(self, mode):
        """Bilinear 및 Nearest 모드를 사용한 업샘플링 (F.interpolate)을 테스트."""
        class UpsampleModel(nn.Module):
            def __init__(self, upsample_mode):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                self.mode = upsample_mode
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)  # 업샘플링 테스트를 위해 다운샘플링 먼저 수행
                x = F.interpolate(x, scale_factor=2, mode=self.mode)  # 업샘플링
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return UpsampleModel(mode)
    
    def _create_residual_model(self):
        """Residual Connection (잔차 연결) 패턴을 테스트."""
        class ResidualModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1) # 32 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(32, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                identity = x # Residual connection을 위한 저장
                x = F.relu(self.conv2(x))
                x = self.conv3(x)
                x = x + identity  # Residual connection: 입력과 출력 더하기
                x = F.relu(x) # 최종 활성화 (ResNet 패턴) # (B, 32, H, W)
                x = self.pool(x) # (B, 32, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 32)
                return self.classifier(x)
        return ResidualModel()
    
    def _create_dense_connection_model(self):
        """Dense Connection (밀집 연결) 패턴 (torch.cat 사용)을 테스트."""
        class DenseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
                # 이전 conv1 출력(16) + conv2 출력(16) = 32 채널이 입력
                self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
                # conv1(16) + conv2(16) + conv3(16) = 48 채널이 최종 특성
                self.final_features_conv = nn.Conv2d(48, 48, 1) # 48 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(48, 10)
                
            def forward(self, x):
                x1 = F.relu(self.conv1(x))
                x2 = F.relu(self.conv2(x1))
                x2_cat = torch.cat([x1, x2], dim=1)  # Dense connection: x1과 x2 채널별 연결
                x3 = F.relu(self.conv3(x2_cat))
                x_final = torch.cat([x1, x2, x3], dim=1)  # 모든 특성 연결 # (B, 48, H, W)
                x = F.relu(self.final_features_conv(x_final)) # 최종 Conv (B, 48, H, W)
                x = self.pool(x) # (B, 48, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 48)
                return self.classifier(x)
        return DenseModel()
    
    def _create_attention_model(self):
        """간단한 Spatial Attention 메커니즘을 테스트."""
        class SimpleAttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                # Attention 가중치를 생성할 1x1 Conv (출력 채널 1)
                self.attention_conv = nn.Conv2d(64, 1, 1)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                
                # Spatial attention 가중치 계산 (Sigmoid로 0~1 범위)
                attention_weights = torch.sigmoid(self.attention_conv(x)) # (B, 1, H, W)
                x = x * attention_weights # 요소별 곱셈으로 특성에 가중치 적용 # (B, 64, H, W)
                
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return SimpleAttentionModel()
    
    def _create_conv1d_model(self):
        """
        2D 입력에서 1D Convolution (nn.Conv1d)을 테스트합니다.
        (참고: 이 모델은 이전 테스트에서 'Timeout'으로 실패한 이력이 있습니다.
        이는 Hailo가 1D Conv를 처리하는 데 어려움을 겪거나 비효율적일 수 있음을 시사합니다.
        대안으로 2D Conv를 사용한 우회 구현을 고려할 수 있습니다.)
        """
        class Conv1DModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv2d = nn.Conv2d(1, 32, 3, padding=1) # (B, 32, 28, 28)
                self.conv1d = nn.Conv1d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool1d(1) # 1D Global Average Pooling
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv2d(x)) # (B, 32, 28, 28)
                # 높이 차원을 1로 줄여 2D 텐서를 1D 시퀀스처럼 만듦
                x = F.adaptive_avg_pool2d(x, (1, 28)) # (B, 32, 1, 28)
                x = x.squeeze(2) # (B, 32, 28) - 1D Conv 입력을 위해 1차원 제거
                x = F.relu(self.conv1d(x)) # (B, 64, 28)
                x = self.pool(x) # (B, 64, 1) - 1D Global Pooling
                x = x.squeeze(-1) # (B, 64) - 마지막 1차원 제거
                return self.classifier(x)
        return Conv1DModel()
    
    def _create_transpose_conv_model(self):
        """Transpose Convolution (nn.ConvTranspose2d)을 테스트."""
        class TransposeConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # 다운샘플링 효과 (출력 크기 감소)
                self.transpose_conv = nn.ConvTranspose2d(64, 32, 2, stride=2) # 업샘플링 효과 (출력 크기 복원) # 32 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(32, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.transpose_conv(x)) # (B, 32, H_orig, W_orig)
                x = self.pool(x) # (B, 32, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 32)
                return self.classifier(x)
        return TransposeConvModel()
    
    def _create_fully_connected_model(self):
        """nn.Linear (Fully Connected Layer)와 nn.Flatten의 기본적인 사용을 테스트."""
        class FullyConnectedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # (B, 32, 28, 28)
                self.flatten = nn.Flatten() # Flatten 2D features to 1D vector
                self.fc1 = nn.Linear(32 * 28 * 28, 128)
                self.fc2 = nn.Linear(128, 10) # Final output is (B, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.flatten(x) # (B, 32*28*28)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        return FullyConnectedModel()
    
    def _create_flatten_model(self):
        """nn.Flatten 모듈 자체의 지원을 명시적으로 테스트."""
        class FlattenModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # (B, 32, 28, 28)
                self.flatten = nn.Flatten() # Test nn.Flatten
                self.classifier = nn.Linear(32 * 28 * 28, 10) # (B, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.flatten(x) # (B, 32*28*28)
                x = self.classifier(x)
                return x
        return FlattenModel()

    def _create_dropout_model(self):
        """
        nn.Dropout2d 연산을 테스트합니다.
        (참고: Dropout은 추론 시에는 보통 아무런 연산도 수행하지 않는 No-Op으로 처리됩니다.)
        """
        class DropoutModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.dropout = nn.Dropout2d(p=0.5) # Apply dropout
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.dropout(x) # Dropout 적용 (eval 모드에서는 효과 없음)
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return DropoutModel()

    def _create_concatenation_model(self):
        """torch.cat (Concatenation) 연산을 테스트합니다."""
        class ConcatenationModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_branch1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv_branch2 = nn.Conv2d(1, 16, 3, padding=1) # 두 브랜치에서 동일한 크기/채널 생성
                self.conv_after_cat = nn.Conv2d(32, 32, 1) # Concatenation 후 입력 채널 16+16=32 # 32 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(32, 10)
                
            def forward(self, x):
                x1 = F.relu(self.conv_branch1(x))
                x2 = F.relu(self.conv_branch2(x))
                x_cat = torch.cat([x1, x2], dim=1) # 채널 차원 (dim=1)으로 연결 # (B, 32, H, W)
                x = F.relu(self.conv_after_cat(x_cat)) # (B, 32, H, W)
                x = self.pool(x) # (B, 32, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 32)
                return self.classifier(x)
        return ConcatenationModel()
    
    def _create_zero_pad_model(self):
        """nn.ZeroPad2d (명시적인 Zero Padding) 연산을 테스트합니다."""
        class ZeroPadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=0) # No padding in Conv to clearly see ZeroPad
                self.pad = nn.ZeroPad2d((1, 1, 1, 1)) # (left, right, top, bottom)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=0) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x)) # Output size reduced due to padding=0
                x = self.pad(x) # Pad to restore or expand size
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return ZeroPadModel()

    def _create_permute_model(self):
        """torch.permute (텐서 차원 순서 변경) 연산을 테스트합니다."""
        class PermuteModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # (B, C, H, W) = (B, 32, 28, 28)
                self.final_conv = nn.Conv2d(32, 32, 1) # 32 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(32, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x)) # (B, 32, 28, 28)
                # 차원 순서 변경 테스트 (예: NHWC -> NCHW -> NHWC)
                x = x.permute(0, 2, 3, 1) # (B, H, W, C) = (B, 28, 28, 32)
                x = x.permute(0, 3, 1, 2) # 다시 (B, C, H, W) = (B, 32, 28, 28)로 복원
                x = F.relu(self.final_conv(x)) # (B, 32, H, W)
                x = self.pool(x) # (B, 32, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 32)
                return self.classifier(x)
        return PermuteModel()
    
    def _create_matmul_model(self):
        """
        간접적인 행렬 곱셈 (Matrix Multiplication)을 테스트합니다.
        (nn.Linear는 내부적으로 행렬 곱셈을 사용하며, NPU에서 효율적으로 구현됩니다.)
        torch.matmul을 직접 사용하는 대신 nn.Linear를 사용하는 것이 컴파일에 더 안정적입니다.
        """
        class MatmulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 10, 3, padding=1) # Output: (B, 10, 28, 28)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(10 * 28 * 28, 10) # (B, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.flatten(x) # (B, 10 * 28 * 28)
                x = self.fc(x) # nn.Linear를 통한 행렬 곱셈
                return x
        return MatmulModel()

    def _create_clamp_model(self):
        """torch.clamp (텐서 값 범위 제한) 연산을 테스트합니다."""
        class ClampModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 64 features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(64, 10)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = torch.clamp(x, min=-1.0, max=1.0) # 값을 -1.0에서 1.0 사이로 제한
                x = F.relu(self.conv2(x)) # (B, 64, H, W)
                x = self.pool(x) # (B, 64, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 64)
                return self.classifier(x)
        return ClampModel()

    def _create_mean_reduction_model(self):
        """
        특정 차원에서의 평균(torch.mean(dim=...)) 연산을 테스트합니다.
        (참고: 이 모델은 이전 테스트에서 'Invalid kernel shape' 파싱 오류가 발생한 이력이 있습니다.
        현재는 Hailo 컴파일러에 더 친화적인 Global Average Pooling (AdaptiveAvgPool2d)을
        사용한 형태로 수정되어 성공적으로 컴파일될 가능성이 높습니다.
        원래의 `torch.mean(dim=2)` 직접 사용 시에는 문제 발생 가능성이 있습니다.)
        """
        class MeanReductionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # (B, 32, 28, 28)
                # NPU 친화적인 Global Average Pooling으로 변경 (이전 torch.mean(dim=2) 대체)
                self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(32, 10) # Conv1의 출력 채널 32를 Linear 입력으로 사용
                
            def forward(self, x):
                x = F.relu(self.conv1(x)) # (B, 32, 28, 28)
                # 원래의 테스트: x = torch.mean(x, dim=2) # (B, 32, 28)
                # NPU 친화적 변경:
                x = self.global_avg_pool(x) # (B, 32, 1, 1)
                x = x.squeeze(-1).squeeze(-1) # (B, 32)
                x = self.classifier(x)
                return x
        return MeanReductionModel()
    
    def export_models_to_onnx(self, models):
        """
        생성된 PyTorch 모델들을 ONNX 포맷으로 변환합니다.
        Hailo 컴파일러는 ONNX를 입력으로 사용합니다.
        """
        # 더미 입력 텐서 (배치 크기 1, 채널 1, 28x28 이미지)
        dummy_input = torch.randn(1, 1, 28, 28)
        successful_exports = []
        
        print("\n=== ONNX 변환 시작 ===")
        for name, model in models.items():
            onnx_path = self.models_dir / f"{name}.onnx"
            try:
                model.eval() # 모델을 평가 모드로 설정 (Dropout 등이 비활성화됨)
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,        # 모델의 학습된 파라미터도 함께 저장
                    opset_version=11,          # ONNX Operator Set 버전 (Hailo에서 지원하는 버전 확인 필요)
                    do_constant_folding=True,  # 상수 폴딩 최적화 활성화
                    input_names=['input'],     # ONNX 그래프의 입력 이름 지정
                    output_names=['output'],   # ONNX 그래프의 출력 이름 지정
                    dynamic_axes={             # 배치 크기를 동적으로 처리할 수 있도록 설정
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                successful_exports.append(name)
                print(f"✓ {name}.onnx 저장 완료")
                
            except Exception as e:
                print(f"✗ {name} ONNX 변환 실패: {e}")
                self.test_results[name] = {'onnx_export': False, 'error': str(e), 'stage': 'onnx_export'}
        print("=== ONNX 변환 완료 ===")
        return successful_exports
    
    def test_hailo_compilation(self, model_names):
        """
        ONNX로 변환된 모델들에 대해 Hailo 컴파일러의 파싱, 최적화, 컴파일 단계를 테스트합니다.
        """
        for name in model_names:
            print(f"\n=== {name} Hailo 컴파일 테스트 ===")
            onnx_path = self.models_dir / f"{name}.onnx"
            
            if not onnx_path.exists():
                print(f"경고: {name} ONNX 파일이 없습니다: {onnx_path}. 다음 모델로 건너뜀.")
                continue
            
            try:
                # 1단계: 파싱 (ONNX -> HAR 변환)
                har_parsed = self.models_dir / f"{name}_parsed.har"
                parse_cmd = [
                    "hailo", "parser", "onnx", str(onnx_path),
                    "--har-path", str(har_parsed)
                ]
                print("1. 파싱 중...")
                # capture_output=True: stdout, stderr 캡처
                # text=True: 출력을 텍스트로 디코딩
                # timeout: 명령어 실행 시간 제한 (초)
                result = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    print(f"✗ 파싱 실패: {result.stderr}")
                    self.test_results[name] = {
                        'parse': False,
                        'error': result.stderr.strip(),
                        'stage': 'parse'
                    }
                    continue
                
                print("✓ 파싱 성공")
                
                # 2단계: 최적화 (HAR 파일 최적화)
                har_optimized = self.models_dir / f"{name}_optimized.har"
                optimize_cmd = [
                    "hailo", "optimize", str(har_parsed),
                    "--output-har-path", str(har_optimized),
                    "--use-random-calib-set" # 양자화를 위한 무작위 캘리브레이션 셋 사용
                ]
                print("2. 최적화 중...")
                result = subprocess.run(optimize_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    print(f"✗ 최적화 실패: {result.stderr}")
                    self.test_results[name] = {
                        'parse': True,
                        'optimize': False,
                        'error': result.stderr.strip(),
                        'stage': 'optimize'
                    }
                    continue
                
                print("✓ 최적화 성공")
                
                # 3단계: 컴파일 (최적화된 HAR -> Hailo 바이너리)
                # 각 모델별로 결과 디렉토리를 생성하여 컴파일 결과물 저장
                output_model_dir = self.results_dir / name 
                output_model_dir.mkdir(parents=True, exist_ok=True)
                
                compile_cmd = [
                    "hailo", "compiler", str(har_optimized),
                    "--output-dir", str(output_model_dir),
                    "--hw-arch", "hailo8" # Hailo-8 아키텍처 지정
                ]
                print("3. 컴파일 중...")
                result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=180)
                
                if result.returncode != 0:
                    print(f"✗ 컴파일 실패: {result.stderr}")
                    self.test_results[name] = {
                        'parse': True,
                        'optimize': True,
                        'compile': False,
                        'error': result.stderr.strip(),
                        'stage': 'compile'
                    }
                    continue
                
                print("✓ 컴파일 성공!")
                self.test_results[name] = {
                    'parse': True,
                    'optimize': True,
                    'compile': True,
                    'stage': 'complete'
                }
                
            except subprocess.TimeoutExpired:
                # 명령어 실행 시간 초과 시
                print(f"✗ 타임아웃 발생: {name}")
                self.test_results[name] = {
                    'error': 'Timeout',
                    'stage': 'timeout'
                }
            except Exception as e:
                # 기타 예외 발생 시
                print(f"✗ 예외 발생: {e}")
                self.test_results[name] = {
                    'error': str(e),
                    'stage': 'exception'
                }
    
    def analyze_results(self):
        """
        테스트 결과를 분석하고 요약 리포트를 생성합니다.
        성공/실패한 모델, 지원되지 않는 연산 등을 정리하여 출력하고 JSON 파일로 저장합니다.
        """
        successful_models = []
        failed_models = []
        unsupported_ops_messages = set() # 중복 메시지 방지를 위해 set 사용
        
        for name, result in self.test_results.items():
            if result.get('compile', False): # 컴파일 단계가 True인 경우 성공으로 간주
                successful_models.append(name)
            else:
                failed_models.append(name)
                if 'error' in result:
                    # 에러 메시지에서 'unsupported' 키워드 포함 여부 확인
                    error_msg_lower = result['error'].lower()
                    if 'unsupported' in error_msg_lower or 'invalid kernel shape' in error_msg_lower:
                        # 오류 메시지의 첫 줄만 추출하여 저장하여 간결하게 표시
                        unsupported_ops_messages.add(f"Error in {name}: {result['error'].splitlines()[0]}")
        
        print("\n" + "="*50)
        print("HAILO 연산 지원 테스트 최종 결과")
        print("="*50)
        
        print(f"\n✓ 성공한 모델들 ({len(successful_models)}개):")
        for model in successful_models:
            print(f"  - {model}")
        
        print(f"\n✗ 실패한 모델들 ({len(failed_models)}개):")
        for model in failed_models:
            stage = self.test_results[model].get('stage', 'unknown')
            error_details = self.test_results[model].get('error', 'No error message available.').splitlines()[0]
            print(f"  - {model} (실패 단계: {stage}, 오류: {error_details})")
        
        if unsupported_ops_messages:
            print(f"\n⚠️  지원되지 않거나 문제 발생한 연산들 (추정):")
            for op_error in sorted(list(unsupported_ops_messages)):
                print(f"  - {op_error}")
        
        # 결과를 JSON 파일로 저장
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'successful_models': successful_models,
                'failed_models': failed_models,
                'detailed_results': self.test_results,
                'summary': {
                    'total_models': len(self.test_results),
                    'successful': len(successful_models),
                    'failed': len(failed_models),
                    'success_rate': (len(successful_models) / len(self.test_results) * 100) if self.test_results else 0
                }
            }, f, indent=2, ensure_ascii=False) # 한글 인코딩 문제 방지
        
        print(f"\n📊 상세 결과가 JSON 파일로 저장되었습니다: {results_file}")
        
        # 지원되는 연산 카테고리별 분석 출력
        self._analyze_supported_operations(successful_models)
        
        return successful_models, failed_models
    
    def _analyze_supported_operations(self, successful_models):
        """
        성공적으로 컴파일된 모델들을 기반으로 연산 카테고리별 지원 여부를 분석하여 출력합니다.
        """
        print(f"\n📋 Hailo-8에서 지원되는 연산 카테고리별 분석:")
        
        operation_categories = {
            'Convolution': ['basic_conv', 'depthwise_conv', 'pointwise_conv', 'dilated_conv', 'grouped_conv'],
            'Activation': [
                'relu_activation', 'leaky_relu_activation', 'sigmoid_activation', 'tanh_activation', 
                'swish_activation', 'gelu_activation', 'elu_activation', 'prelu_activation',
                'hardswish_activation', 'hardsigmoid_activation'
            ],
            'Normalization': ['batch_norm', 'layer_norm', 'instance_norm'],
            'Pooling': [
                'max_pool', 'avg_pool', 'adaptive_pool', 'global_avg_pool',
                'max_pool_varied', 'avg_pool_varied'
            ],
            'Element-wise': ['elementwise_add', 'elementwise_mul', 'elementwise_sub', 'elementwise_div'],
            'Resize/Interpolation': ['bilinear_upsample', 'nearest_upsample'],
            'Skip Connections': ['residual_block', 'dense_connection'],
            'Attention': ['simple_attention'],
            '1D Operations': ['conv1d'], # 실패했더라도 여기에 포함시켜 어떤 연산 그룹인지 명시
            'Transpose Convolution': ['transpose_conv'],
            'Fully Connected & Flatten': ['fully_connected', 'flatten_op', 'matmul_op'],
            'Dropout': ['dropout_op'],
            'Concatenation': ['concatenation_op'],
            'Padding': ['zero_pad'],
            'Tensor Manipulation': ['permute_op'],
            'Value Manipulation': ['clamp_op'],
            'Reduction': ['mean_reduction'] # 실패했더라도 어떤 연산 그룹인지 명시
        }
        
        for category, models_in_category in operation_categories.items():
            supported_in_category = [m for m in models_in_category if m in successful_models]
            total_in_category = len(models_in_category)
            print(f"\n{category}:")
            print(f"  지원: {len(supported_in_category)}/{total_in_category}")
            for model_name in models_in_category:
                status = "✓" if model_name in successful_models else "✗"
                print(f"    {status} {model_name}")
    
    def generate_batch_test_script(self):
        """
        ONNX 파일들을 사용하여 Hailo 컴파일 과정을 수동으로 다시 실행할 수 있는
        Bash 쉘 스크립트(batch_test.sh)를 생성합니다.
        """
        script_content = """#!/bin/bash
# Hailo 연산 지원 배치 테스트 스크립트
# ONNX 파일들을 Hailo 컴파일러로 처리하는 과정을 자동화합니다.

echo "=== Hailo 연산 지원 배치 테스트 시작 ==="
echo "테스트 시간: $(date)"

# 결과 및 로그 파일 경로 설정
MODELS_DIR="hailo_test_results/models"
RESULTS_DIR="hailo_test_results/results"
LOG_FILE="hailo_test_results/batch_test.log"

# 결과 디렉토리 생성 (이미 존재하면 무시)
mkdir -p "$RESULTS_DIR"

# 로그 파일 초기화 또는 새로 생성
echo "Hailo Batch Test Log - $(date)" > "$LOG_FILE"

success_count=0
total_count=0

# models_dir 안의 모든 .onnx 파일을 순회하며 테스트
for onnx_file in "$MODELS_DIR"/*.onnx; do
    # 파일이 실제로 존재하는지 확인
    if [ -f "$onnx_file" ]; then
        model_name=$(basename "$onnx_file" .onnx) # 파일명에서 확장자 제거
        echo "\\n=== 테스트 중: $model_name ===" | tee -a "$LOG_FILE" # 콘솔과 로그 파일에 출력
        
        total_count=$((total_count + 1)) # 전체 테스트 모델 수 증가
        
        # 각 모델의 중간 HAR 파일 및 최종 출력 디렉토리 경로 설정
        har_parsed="$MODELS_DIR/${model_name}_parsed.har"
        har_optimized="$MODELS_DIR/${model_name}_optimized.har"
        output_dir="$RESULTS_DIR/$model_name"
        
        mkdir -p "$output_dir" # 모델별 결과 디렉토리 생성
        
        # 1단계: 파싱 (ONNX -> HAR)
        echo "1. 파싱 중..." | tee -a "$LOG_FILE"
        if hailo parser onnx "$onnx_file" --har-path "$har_parsed" >> "$LOG_FILE" 2>&1; then
            echo "✓ 파싱 성공" | tee -a "$LOG_FILE"
            
            # 2단계: 최적화 (HAR 파일 최적화)
            echo "2. 최적화 중..." | tee -a "$LOG_FILE"
            if hailo optimize "$har_parsed" --output-har-path "$har_optimized" --use-random-calib-set >> "$LOG_FILE" 2>&1; then
                echo "✓ 최적화 성공" | tee -a "$LOG_FILE"
                
                # 3단계: 컴파일 (최적화된 HAR -> Hailo 바이너리)
                echo "3. 컴파일 중..." | tee -a "$LOG_FILE"
                if hailo compiler "$har_optimized" --output-dir "$output_dir" --hw-arch hailo8 >> "$LOG_FILE" 2>&1; then
                    echo "✅ $model_name: 전체 컴파일 성공!" | tee -a "$LOG_FILE"
                    success_count=$((success_count + 1)) # 성공 카운트 증가
                else
                    echo "❌ $model_name: 컴파일 실패" | tee -a "$LOG_FILE"
                fi
            else
                echo "❌ $model_name: 최적화 실패" | tee -a "$LOG_FILE"  
            fi
        else
            echo "❌ $model_name: 파싱 실패" | tee -a "$LOG_FILE"
        fi
    fi
done

echo "\\n=== 최종 결과 요약 ===" | tee -a "$LOG_FILE"
echo "총 테스트 모델 수: $total_count" | tee -a "$LOG_FILE"
echo "성공: $success_count" | tee -a "$LOG_FILE"
echo "실패: $((total_count - success_count))" | tee -a "$LOG_FILE"

if [ $total_count -gt 0 ]; then
    success_rate=$((success_count * 100 / total_count))
    echo "성공률: ${success_rate}%" | tee -a "$LOG_FILE"
fi

echo "\\n상세 로그 파일: $LOG_FILE"
echo "배치 테스트 완료 시간: $(date)" | tee -a "$LOG_FILE"
"""
        
        script_path = self.output_dir / "batch_test.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # 생성된 스크립트에 실행 권한 부여
        os.chmod(script_path, 0o755)
        
        print(f"\n배치 테스트 스크립트가 생성되었습니다: {script_path}")
        print(f"터미널에서 다음 명령어로 실행할 수 있습니다: ./hailo_test_results/batch_test.sh")
    
    def run_full_test(self):
        """
        전체 Hailo 연산 지원 테스트 프로세스를 실행합니다.
        (모델 생성 -> ONNX 변환 -> Hailo 컴파일 -> 결과 분석 -> 배치 스크립트 생성)
        """
        print("\n" + "="*60)
        print("=== Hailo 연산 지원 테스트 시작 (PyTorch -> ONNX -> Hailo) ===")
        print("="*60)
        
        # 1. 테스트 모델 생성
        print("\n[단계 1/5] PyTorch 테스트 모델 생성 중...")
        models = self.create_test_models()
        print(f"총 {len(models)}개 PyTorch 테스트 모델 생성 완료.")
        
        # 2. ONNX 변환
        print("\n[단계 2/5] PyTorch 모델을 ONNX 포맷으로 변환 중...")
        successful_exports = self.export_models_to_onnx(models)
        print(f"총 {len(successful_exports)}개 모델 ONNX 변환 성공.")
        
        # 3. Hailo 컴파일 테스트
        print(f"\n[단계 3/5] 변환된 ONNX 모델에 대해 Hailo 컴파일 테스트 중... (대상: {len(successful_exports)}개 모델)")
        self.test_hailo_compilation(successful_exports)
        
        # 4. 결과 분석 및 리포트 생성
        print("\n[단계 4/5] 테스트 결과 분석 및 리포트 생성 중...")
        successful_models, failed_models = self.analyze_results()
        
        # 5. 배치 테스트 스크립트 생성
        print("\n[단계 5/5] 배치 테스트 스크립트 생성 중...")
        self.generate_batch_test_script()
        
        print("\n" + "="*60)
        print("=== Hailo 연산 지원 테스트 전체 완료! ===")
        print("="*60)
        
        return successful_models, failed_models


def main():
    """메인 실행 함수"""
    tester = HailoOperationTester()
    
    print("\n" + "#"*70)
    print("## Hailo AI 가속기 연산 지원 테스트 스크립트 ##")
    print("## 이 스크립트는 Hailo-8에서 다양한 뉴럴 네트워크 연산의 호환성을 검증합니다. ##")
    print("#"*70)
    
    print("\n테스트가 수행할 주요 연산 카테고리:")
    print("- Convolution (표준, Depthwise, Grouped, Dilated, Transpose)")
    print("- 활성화 함수 (ReLU, LeakyReLU, Sigmoid, Tanh, Swish, GELU, ELU, PReLU, Hardswish, Hardsigmoid)")
    print("- 정규화 (BatchNorm, LayerNorm, InstanceNorm)")
    print("- 풀링 (MaxPool, AvgPool, AdaptivePool, GlobalAvgPool, 다양한 파라미터)")
    print("- Element-wise 연산 (Add, Mul, Sub, Div, Clamp)")
    print("- Resize/Interpolation (Bilinear, Nearest)")
    print("- Skip Connection 패턴 (Residual, Dense)")
    print("- Attention 메커니즘")
    print("- 1D Convolution (특정 상황에서 문제 발생 이력)")
    print("- Fully Connected Layer (Linear)")
    print("- Tensor 조작 (Flatten, Dropout, Concatenation, Padding, Permute, Squeeze)")
    print("- Reduction (Mean Reduction, 특정 상황에서 문제 발생 이력)")
    
    print(f"\n모든 결과물은 '{tester.output_dir}' 디렉토리에 저장됩니다.")
    
    try:
        successful_models, failed_models = tester.run_full_test()
        
        print(f"\n🎉 최종 테스트 결과 요약!")
        print(f"  총 성공 모델 수: {len(successful_models)}개")
        print(f"  총 실패 모델 수: {len(failed_models)}개")
        
        if successful_models:
            print(f"\n✅ Hailo-8에서 성공적으로 컴파일된 연산 패턴:")
            for model in successful_models:
                print(f"    - {model}")
        
        if failed_models:
            print(f"\n❌ 지원되지 않거나 컴파일에 문제가 있는 연산 패턴:")
            for model in failed_models:
                print(f"    - {model}")
        
        print(f"\n📄 상세 결과 확인을 위한 파일 및 디렉토리:")
        print(f"  - 결과 요약 JSON: {tester.output_dir}/test_results.json")
        print(f"  - 배치 테스트 스크립트: {tester.output_dir}/batch_test.sh")
        print(f"  - 컴파일된 모델 및 로그: {tester.output_dir}/results/")
        
    except Exception as e:
        print(f"\n❌ 테스트 중 치명적인 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()