"""
Multi-Model Workload Analyzer
Analyzes different types of neural networks and their ISA optimization potential
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from collections import defaultdict
import time

class WorkloadProfiler:
    """Analyzes different types of neural network workloads"""
    
    def __init__(self):
        self.model_types = {
            'CNN': self._create_cnn_model,
            'Transformer': self._create_transformer_model,
            'SNN': self._create_snn_model,
            'TinyML': self._create_tinyml_model,
            'GAN': self._create_gan_model
        }
        self.profiling_results = {}
    
    def analyze_all_workloads(self, input_sizes: Dict[str, Tuple] = None) -> Dict[str, Any]:
        """Analyze all supported workload types"""
        if input_sizes is None:
            input_sizes = {
                'CNN': (1, 3, 224, 224),
                'Transformer': (1, 512, 768),
                'SNN': (1, 784),
                'TinyML': (1, 28, 28),
                'GAN': (1, 100)
            }
        
        results = {}
        
        for model_type in self.model_types:
            print(f"Analyzing {model_type} workload...")
            try:
                model = self.model_types[model_type]()
                input_size = input_sizes.get(model_type, (1, 3, 224, 224))
                
                profile_data = self._profile_model(model, input_size, model_type)
                isa_recommendations = self._analyze_isa_opportunities(profile_data, model_type)
                
                results[model_type] = {
                    'profile_data': profile_data,
                    'isa_recommendations': isa_recommendations,
                    'optimization_potential': self._calculate_optimization_potential(profile_data)
                }
                
            except Exception as e:
                print(f"Error analyzing {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def _create_cnn_model(self) -> nn.Module:
        """Create CNN model for analysis"""
        return nn.Sequential(
            # Convolutional layers - heavy on conv2d operations
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling and classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1000)
        )
    
    def _create_transformer_model(self) -> nn.Module:
        """Create Transformer model for analysis"""
        class MultiHeadAttention(nn.Module):
            def __init__(self, d_model=768, n_heads=12):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_k = d_model // n_heads
                
                self.w_q = nn.Linear(d_model, d_model)
                self.w_k = nn.Linear(d_model, d_model)
                self.w_v = nn.Linear(d_model, d_model)
                self.w_o = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                batch_size, seq_len, _ = x.size()
                
                # Linear transformations - heavy on matrix multiplications
                q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
                
                # Attention computation - batch matrix multiplications
                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                
                # Output projection
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
                return self.w_o(out)
        
        class TransformerBlock(nn.Module):
            def __init__(self, d_model=768):
                super().__init__()
                self.attention = MultiHeadAttention(d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
                
            def forward(self, x):
                # Self-attention with residual connection
                attn_out = self.attention(x)
                x = self.norm1(x + attn_out)
                
                # Feed-forward with residual connection
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                
                return x
        
        return nn.Sequential(
            nn.Embedding(30000, 768),  # Vocabulary embedding
            *[TransformerBlock() for _ in range(6)],  # 6 transformer blocks
            nn.LayerNorm(768),
            nn.Linear(768, 30000)  # Output projection
        )
    
    def _create_snn_model(self) -> nn.Module:
        """Create Spiking Neural Network model for analysis"""
        class SpikingLayer(nn.Module):
            def __init__(self, input_size, output_size, threshold=1.0):
                super().__init__()
                self.linear = nn.Linear(input_size, output_size)
                self.threshold = threshold
                self.membrane_potential = None
                
            def forward(self, x):
                batch_size = x.size(0)
                if self.membrane_potential is None:
                    self.membrane_potential = torch.zeros(batch_size, self.linear.out_features)
                
                # Sparse operations - key characteristic of SNNs
                current = self.linear(x)
                self.membrane_potential += current
                
                # Spiking mechanism - threshold activation
                spikes = (self.membrane_potential >= self.threshold).float()
                self.membrane_potential[self.membrane_potential >= self.threshold] = 0
                
                return spikes
        
        return nn.Sequential(
            SpikingLayer(784, 256),
            SpikingLayer(256, 128),
            SpikingLayer(128, 64),
            SpikingLayer(64, 10)
        )
    
    def _create_tinyml_model(self) -> nn.Module:
        """Create TinyML model for analysis (memory-constrained)"""
        return nn.Sequential(
            # Extremely small model for edge devices
            nn.Flatten(),
            nn.Linear(28*28, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
    
    def _create_gan_model(self) -> nn.Module:
        """Create GAN generator model for analysis"""
        return nn.Sequential(
            # Generator network with transposed convolutions
            nn.Linear(100, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def _profile_model(self, model: nn.Module, input_size: Tuple, model_type: str) -> Dict[str, Any]:
        """Profile a specific model"""
        model.eval()
        layer_times = {}
        operation_counts = defaultdict(int)
        memory_usage = {}
        
        # Hook function to measure layer execution times
        def hook_fn(module, input, output, name):
            # Simulate timing (in real implementation, use actual timing)
            if isinstance(module, nn.Conv2d):
                # Convolution timing based on parameters
                params = sum(p.numel() for p in module.parameters())
                layer_times[name] = params * 1e-8  # Simulated time
                operation_counts['conv2d'] += 1
            elif isinstance(module, nn.Linear):
                params = sum(p.numel() for p in module.parameters())
                layer_times[name] = params * 5e-9  # Matrix mult is faster per param
                operation_counts['linear'] += 1
            elif isinstance(module, (nn.ReLU, nn.GELU)):
                if hasattr(output, 'numel'):
                    layer_times[name] = output.numel() * 1e-10
                else:
                    layer_times[name] = 1e-6
                operation_counts['activation'] += 1
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(output, 'numel'):
                    layer_times[name] = output.numel() * 2e-10
                else:
                    layer_times[name] = 1e-6
                operation_counts['normalization'] += 1
            elif isinstance(module, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                if hasattr(output, 'numel'):
                    layer_times[name] = output.numel() * 5e-11
                else:
                    layer_times[name] = 1e-7
                operation_counts['pooling'] += 1
            else:
                layer_times[name] = 1e-6  # Default timing
                operation_counts['other'] += 1
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)
        
        # Run forward pass
        dummy_input = torch.randn(input_size)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(dummy_input)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        
        # Calculate memory usage (rough estimate)
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = total_params * 4  # 4 bytes per float32
        
        if hasattr(output, 'numel'):
            activation_memory = output.numel() * 4
        else:
            activation_memory = 1000 * 4  # Default estimate
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Process results
        total_layer_time = sum(layer_times.values())
        layers = []
        
        for name, time_val in layer_times.items():
            percentage = (time_val / total_layer_time * 100) if total_layer_time > 0 else 0
            
            # Get layer info
            layer_module = dict(model.named_modules()).get(name)
            layer_type = type(layer_module).__name__ if layer_module else 'Unknown'
            
            params = sum(p.numel() for p in layer_module.parameters()) if layer_module else 0
            
            layers.append({
                'name': name,
                'type': layer_type,
                'avg_time': time_val,
                'percentage': percentage,
                'parameters': params,
                'flops_estimate': self._estimate_flops(layer_module, dummy_input) if layer_module else 0
            })
        
        # Sort by execution time
        layers.sort(key=lambda x: x['avg_time'], reverse=True)
        
        return {
            'total_time': total_time,
            'layers': layers,
            'model_type': model_type,
            'operation_counts': dict(operation_counts),
            'memory_usage': {
                'parameters': param_memory,
                'activations': activation_memory,
                'total': param_memory + activation_memory
            },
            'bottlenecks': [layer for layer in layers if layer['percentage'] > 10]
        }
    
    def _estimate_flops(self, module, input_tensor) -> int:
        """Estimate FLOPs for a layer"""
        if isinstance(module, nn.Conv2d):
            # For conv2d: output_elements * kernel_elements
            if hasattr(input_tensor, 'shape'):
                batch, in_c, in_h, in_w = input_tensor.shape
                out_h = (in_h + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                out_w = (in_w + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * in_c
                return batch * out_h * out_w * module.out_channels * kernel_flops
        elif isinstance(module, nn.Linear):
            # For linear: input_features * output_features
            return module.in_features * module.out_features
        elif isinstance(module, (nn.ReLU, nn.GELU)):
            # For activations: number of elements
            return input_tensor.numel() if hasattr(input_tensor, 'numel') else 1000
        
        return 1000  # Default estimate
    
    def _analyze_isa_opportunities(self, profile_data: Dict[str, Any], model_type: str) -> List[Dict[str, Any]]:
        """Analyze ISA optimization opportunities for specific model type"""
        opportunities = []
        operation_counts = profile_data.get('operation_counts', {})
        bottlenecks = profile_data.get('bottlenecks', [])
        
        # Model-specific ISA recommendations
        if model_type == 'CNN':
            # CNN-specific optimizations
            if operation_counts.get('conv2d', 0) > 0:
                opportunities.append({
                    'instruction': 'VCONV.FUSED',
                    'description': 'Fused convolution-batch norm-activation',
                    'target_operations': ['conv2d', 'batch_norm', 'relu'],
                    'estimated_speedup': 2.8,
                    'rationale': 'CNNs have many conv-bn-relu patterns that can be fused'
                })
            
            if operation_counts.get('pooling', 0) > 0:
                opportunities.append({
                    'instruction': 'VPOOL.ADAPTIVE',
                    'description': 'Adaptive pooling with multiple kernel sizes',
                    'target_operations': ['pooling'],
                    'estimated_speedup': 1.8,
                    'rationale': 'Pooling operations can be accelerated with custom hardware'
                })
        
        elif model_type == 'Transformer':
            # Transformer-specific optimizations
            if operation_counts.get('linear', 0) > 0:
                opportunities.append({
                    'instruction': 'MHATTN.BLOCK',
                    'description': 'Multi-head attention block acceleration',
                    'target_operations': ['linear', 'softmax'],
                    'estimated_speedup': 3.5,
                    'rationale': 'Transformers are dominated by matrix multiplications in attention'
                })
                
                opportunities.append({
                    'instruction': 'SOFTMAX.V',
                    'description': 'Vectorized softmax with numerical stability',
                    'target_operations': ['softmax'],
                    'estimated_speedup': 2.2,
                    'rationale': 'Attention mechanism requires many softmax operations'
                })
        
        elif model_type == 'SNN':
            # Spiking Neural Network optimizations
            opportunities.append({
                'instruction': 'SPIKE.THRESH',
                'description': 'Threshold-based spiking with membrane potential',
                'target_operations': ['threshold', 'accumulate'],
                'estimated_speedup': 4.1,
                'rationale': 'SNNs have unique sparse, event-driven computation patterns'
            })
            
            opportunities.append({
                'instruction': 'SPARSE.MADD',
                'description': 'Sparse multiply-accumulate for spike trains',
                'target_operations': ['sparse_linear'],
                'estimated_speedup': 3.2,
                'rationale': 'SNNs operate on sparse spike data'
            })
        
        elif model_type == 'TinyML':
            # TinyML-specific optimizations
            opportunities.append({
                'instruction': 'QUANT.MADD',
                'description': 'Quantized multiply-accumulate for 8-bit/4-bit',
                'target_operations': ['linear', 'conv2d'],
                'estimated_speedup': 2.1,
                'rationale': 'TinyML models use aggressive quantization'
            })
            
            opportunities.append({
                'instruction': 'MEMOPT.LOAD',
                'description': 'Memory-optimized loading for constrained devices',
                'target_operations': ['memory_access'],
                'estimated_speedup': 1.6,
                'rationale': 'TinyML is heavily memory-constrained'
            })
        
        elif model_type == 'GAN':
            # GAN-specific optimizations
            opportunities.append({
                'instruction': 'CONVT.FUSED',
                'description': 'Fused transposed convolution-batch norm-activation',
                'target_operations': ['conv_transpose', 'batch_norm', 'activation'],
                'estimated_speedup': 2.7,
                'rationale': 'GANs use many transposed convolutions for upsampling'
            })
        
        # Add general optimizations based on bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'Conv2d' and bottleneck['percentage'] > 20:
                opportunities.append({
                    'instruction': f"VCONV.{bottleneck['name'].upper()}",
                    'description': f"Specialized convolution for {bottleneck['name']}",
                    'target_operations': ['conv2d'],
                    'estimated_speedup': 2.5,
                    'rationale': f"Layer {bottleneck['name']} is a major bottleneck at {bottleneck['percentage']:.1f}%"
                })
        
        return opportunities
    
    def _calculate_optimization_potential(self, profile_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall optimization potential"""
        operation_counts = profile_data.get('operation_counts', {})
        bottlenecks = profile_data.get('bottlenecks', [])
        
        # Calculate bottleneck percentage
        bottleneck_time = sum(layer['percentage'] for layer in bottlenecks)
        
        # Estimate potential speedup based on operation types
        conv_ops = operation_counts.get('conv2d', 0)
        linear_ops = operation_counts.get('linear', 0)
        activation_ops = operation_counts.get('activation', 0)
        
        max_theoretical_speedup = 1.0
        if conv_ops > 0:
            max_theoretical_speedup *= 1.5  # 50% improvement from conv optimizations
        if linear_ops > 0:
            max_theoretical_speedup *= 1.4  # 40% improvement from matmul optimizations
        if activation_ops > 0:
            max_theoretical_speedup *= 1.2  # 20% improvement from activation optimizations
        
        return {
            'bottleneck_percentage': bottleneck_time,
            'max_theoretical_speedup': max_theoretical_speedup,
            'isa_extension_potential': min(bottleneck_time / 100.0 * max_theoretical_speedup, 0.8),
            'memory_optimization_potential': 0.3 if profile_data.get('memory_usage', {}).get('total', 0) > 1e6 else 0.1
        }
    
    def generate_comparative_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comparative analysis report"""
        report = "Multi-Model Workload Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Summary table
        report += "WORKLOAD COMPARISON SUMMARY:\n"
        report += f"{'Model Type':<15} {'Bottlenecks':<12} {'Max Speedup':<12} {'ISA Potential':<14}\n"
        report += "-" * 55 + "\n"
        
        for model_type, results in analysis_results.items():
            if 'error' in results:
                continue
                
            opt_potential = results['optimization_potential']
            bottlenecks = len(results['profile_data']['bottlenecks'])
            max_speedup = opt_potential['max_theoretical_speedup']
            isa_potential = opt_potential['isa_extension_potential']
            
            report += f"{model_type:<15} {bottlenecks:<12} {max_speedup:<12.2f} {isa_potential:<14.2f}\n"
        
        report += "\n"
        
        # Detailed analysis for each model
        for model_type, results in analysis_results.items():
            if 'error' in results:
                report += f"\n{model_type.upper()} ANALYSIS: Error - {results['error']}\n"
                continue
            
            report += f"\n{model_type.upper()} DETAILED ANALYSIS:\n"
            report += "-" * 30 + "\n"
            
            profile = results['profile_data']
            recommendations = results['isa_recommendations']
            
            # Operation breakdown
            op_counts = profile.get('operation_counts', {})
            report += f"Operation Types: {dict(op_counts)}\n"
            report += f"Total Execution Time: {profile['total_time']:.4f}s\n"
            report += f"Memory Usage: {profile.get('memory_usage', {}).get('total', 0) / 1024 / 1024:.2f} MB\n"
            
            # Top bottlenecks
            bottlenecks = profile.get('bottlenecks', [])
            if bottlenecks:
                report += f"\nTop Bottlenecks:\n"
                for layer in bottlenecks[:3]:
                    report += f"  • {layer['name']} ({layer['type']}): {layer['percentage']:.1f}% of time\n"
            
            # ISA recommendations
            if recommendations:
                report += f"\nISA Extension Recommendations:\n"
                for rec in recommendations[:3]:
                    report += f"  • {rec['instruction']}: {rec['estimated_speedup']:.1f}x speedup\n"
                    report += f"    Target: {', '.join(rec['target_operations'])}\n"
            
            report += "\n"
        
        return report
    
    def export_results(self, analysis_results: Dict[str, Any], filename: str = "workload_analysis.json"):
        """Export analysis results to JSON"""
        with open(filename, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"Analysis results exported to {filename}")

def run_multi_model_analysis():
    """Run comprehensive multi-model workload analysis"""
    profiler = WorkloadProfiler()
    
    print("Starting multi-model workload analysis...")
    print("This will analyze CNN, Transformer, SNN, TinyML, and GAN workloads")
    
    # Run analysis
    results = profiler.analyze_all_workloads()
    
    # Generate report
    report = profiler.generate_comparative_report(results)
    print("\n" + report)
    
    # Export results
    profiler.export_results(results)
    
    return results

if __name__ == "__main__":
    results = run_multi_model_analysis()