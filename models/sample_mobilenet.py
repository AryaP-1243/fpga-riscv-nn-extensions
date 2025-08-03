"""
Sample MobileNet Model Generator
Creates a simplified MobileNet-like model for profiling demonstrations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import onnx
import torch.onnx

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block used in MobileNet"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, 
            stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class SimpleMobileNet(nn.Module):
    """Simplified MobileNet for profiling purposes"""
    
    def __init__(self, num_classes: int = 1000, width_multiplier: float = 1.0):
        super(SimpleMobileNet, self).__init__()
        
        # Calculate channel sizes based on width multiplier
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            return new_v
        
        # Initial convolution
        input_channels = make_divisible(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        
        # MobileNet blocks configuration: (out_channels, stride)
        self.block_configs = [
            (64, 1),   # Block 1
            (128, 2),  # Block 2
            (128, 1),  # Block 3
            (256, 2),  # Block 4
            (256, 1),  # Block 5
            (512, 2),  # Block 6
            (512, 1),  # Block 7
            (512, 1),  # Block 8
            (512, 1),  # Block 9
            (512, 1),  # Block 10
            (1024, 2), # Block 11
            (1024, 1), # Block 12
        ]
        
        # Build depthwise separable blocks
        self.blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i, (out_channels, stride) in enumerate(self.block_configs):
            out_channels = make_divisible(out_channels * width_multiplier)
            block = DepthwiseSeparableConv(in_channels, out_channels, stride)
            self.blocks.append(block)
            in_channels = out_channels
        
        # Global average pooling and classifier
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable blocks
        for block in self.blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class TinyYOLO(nn.Module):
    """Simplified YOLO-like model for object detection profiling"""
    
    def __init__(self, num_classes: int = 80, num_anchors: int = 3):
        super(TinyYOLO, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Detection head
        output_channels = num_anchors * (5 + num_classes)  # (x, y, w, h, confidence) + classes
        self.conv_detect = nn.Conv2d(256, output_channels, 1, 1, 0)
        
        self._initialize_weights()
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool5(x)
        
        # Detection
        x = self.conv_detect(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def create_sample_models() -> dict:
    """Create sample models for profiling"""
    models = {
        'mobilenet_v2_1.0': SimpleMobileNet(num_classes=1000, width_multiplier=1.0),
        'mobilenet_v2_0.5': SimpleMobileNet(num_classes=1000, width_multiplier=0.5),
        'tiny_yolo': TinyYOLO(num_classes=80)
    }
    
    return models

def export_to_onnx(model: nn.Module, model_name: str, input_size: Tuple[int, ...] = (1, 3, 224, 224)):
    """Export PyTorch model to ONNX format"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size)
    
    # Export to ONNX
    output_path = f"{model_name}.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    return output_path

def get_model_info(model: nn.Module) -> dict:
    """Get detailed information about a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    # Count different layer types
    layer_counts = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if layer_type != type(model).__name__:  # Skip the model itself
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'layer_counts': layer_counts
    }

def benchmark_model(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224), 
                   num_runs: int = 100) -> dict:
    """Benchmark model inference time"""
    import time
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Warm up
    dummy_input = torch.randn(input_size).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        'device': str(device)
    }

if __name__ == "__main__":
    # Create and test sample models
    models = create_sample_models()
    
    for name, model in models.items():
        print(f"\n=== {name.upper()} ===")
        
        # Get model info
        info = get_model_info(model)
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Model size: {info['model_size_mb']:.2f} MB")
        print(f"Layer types: {info['layer_counts']}")
        
        # Benchmark
        input_size = (1, 3, 224, 224) if 'yolo' not in name else (1, 3, 416, 416)
        benchmark = benchmark_model(model, input_size, num_runs=10)
        print(f"Inference time: {benchmark['mean_time']*1000:.2f}±{benchmark['std_time']*1000:.2f} ms")
        
        # Export to ONNX
        try:
            onnx_path = export_to_onnx(model, name, input_size)
            print(f"Exported to: {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
