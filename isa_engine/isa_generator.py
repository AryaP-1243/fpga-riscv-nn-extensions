"""
AI-Powered ISA Extension Generator
Generates custom RISC-V instruction suggestions based on profiling data
"""

import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np

class ISAGenerator:
    """Generate custom RISC-V instruction set extensions based on profiling data"""
    
    def __init__(self):
        self.instruction_templates = self._load_instruction_templates()
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_instruction_templates(self):
        """Load predefined instruction templates for different operations"""
        return {
            'conv2d': {
                'base_name': 'VCONV',
                'description': 'Vectorized 2D Convolution',
                'operands': ['rs1', 'rs2', 'rd'],
                'category': 'neural_compute',
                'base_opcode': '0x7B',
                'variants': [
                    {'suffix': '8', 'desc': '8-bit integer convolution'},
                    {'suffix': '16', 'desc': '16-bit integer convolution'},
                    {'suffix': 'F', 'desc': 'floating-point convolution'}
                ]
            },
            'matmul': {
                'base_name': 'VMMUL',
                'description': 'Vector-Matrix Multiply',
                'operands': ['rs1', 'rs2', 'rd'],
                'category': 'neural_compute',
                'base_opcode': '0x7C',
                'variants': [
                    {'suffix': '8', 'desc': '8-bit matrix multiply'},
                    {'suffix': '16', 'desc': '16-bit matrix multiply'},
                    {'suffix': 'F', 'desc': 'floating-point matrix multiply'}
                ]
            },
            'relu': {
                'base_name': 'RELU',
                'description': 'ReLU Activation Function',
                'operands': ['rs1', 'rd'],
                'category': 'activation',
                'base_opcode': '0x7D',
                'variants': [
                    {'suffix': 'V', 'desc': 'vectorized ReLU'},
                    {'suffix': '6', 'desc': 'ReLU6 with clipping'},
                    {'suffix': 'L', 'desc': 'Leaky ReLU'}
                ]
            },
            'batchnorm': {
                'base_name': 'BNORM',
                'description': 'Batch Normalization',
                'operands': ['rs1', 'rs2', 'rs3', 'rd'],
                'category': 'normalization',
                'base_opcode': '0x7E',
                'variants': [
                    {'suffix': '2D', 'desc': '2D batch normalization'},
                    {'suffix': '1D', 'desc': '1D batch normalization'}
                ]
            },
            'pooling': {
                'base_name': 'VPOOL',
                'description': 'Vectorized Pooling Operation',
                'operands': ['rs1', 'rd'],
                'category': 'pooling',
                'base_opcode': '0x7F',
                'variants': [
                    {'suffix': 'MAX', 'desc': 'max pooling'},
                    {'suffix': 'AVG', 'desc': 'average pooling'},
                    {'suffix': 'GAP', 'desc': 'global average pooling'}
                ]
            }
        }
    
    def _load_optimization_rules(self):
        """Load rules for determining when to suggest ISA extensions"""
        return {
            'time_threshold': 5.0,  # Suggest extension if layer takes >5% of total time
            'flops_threshold': 1000000,  # Suggest for operations with >1M FLOPs
            'parameter_threshold': 10000,  # Consider layers with >10K parameters
            'bottleneck_priority': ['conv2d', 'linear', 'matmul', 'relu', 'batchnorm']
        }
    
    def generate_extensions(self, profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate ISA extension suggestions based on profiling data
        
        Args:
            profile_data: Profiling results from ModelProfiler
            
        Returns:
            List of suggested ISA extensions
        """
        extensions = []
        
        try:
            # Analyze bottlenecks
            bottlenecks = profile_data.get('bottlenecks', [])
            if not bottlenecks:
                # If no explicit bottlenecks, use top time-consuming layers
                layers = profile_data.get('layers', [])
                bottlenecks = [layer for layer in layers[:5] 
                              if layer['percentage'] > self.optimization_rules['time_threshold']]
            
            # Generate extensions for each bottleneck
            for layer in bottlenecks:
                layer_extensions = self._generate_layer_extensions(layer)
                extensions.extend(layer_extensions)
            
            # Remove duplicates and sort by priority
            extensions = self._deduplicate_extensions(extensions)
            extensions = self._prioritize_extensions(extensions, profile_data)
            
            return extensions
            
        except Exception as e:
            raise Exception(f"Error generating ISA extensions: {str(e)}")
    
    def _generate_layer_extensions(self, layer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ISA extensions for a specific layer"""
        extensions = []
        layer_type = layer['type'].lower()
        
        # Map layer types to instruction templates
        if 'conv' in layer_type:
            template = self.instruction_templates['conv2d']
            extensions.extend(self._create_extensions_from_template(template, layer))
        elif 'linear' in layer_type or 'gemm' in layer_type:
            template = self.instruction_templates['matmul']
            extensions.extend(self._create_extensions_from_template(template, layer))
        elif 'relu' in layer_type:
            template = self.instruction_templates['relu']
            extensions.extend(self._create_extensions_from_template(template, layer))
        elif 'batchnorm' in layer_type:
            template = self.instruction_templates['batchnorm']
            extensions.extend(self._create_extensions_from_template(template, layer))
        elif 'pool' in layer_type:
            template = self.instruction_templates['pooling']
            extensions.extend(self._create_extensions_from_template(template, layer))
        
        return extensions
    
    def _create_extensions_from_template(self, template: Dict[str, Any], 
                                       layer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create specific ISA extensions from a template"""
        extensions = []
        
        for variant in template['variants']:
            extension = {
                'name': f"{template['base_name']}.{variant['suffix']}",
                'description': f"{template['description']} - {variant['desc']}",
                'category': template['category'],
                'operands': template['operands'].copy(),
                'opcode': self._generate_opcode(template['base_opcode'], variant['suffix']),
                'target_layer': layer['name'],
                'target_operation': layer['type'],
                'estimated_speedup': self._estimate_speedup(layer, template['category']),
                'instruction_reduction': self._estimate_instruction_reduction(layer, template['category']),
                'assembly_example': self._generate_assembly_example(
                    f"{template['base_name']}.{variant['suffix']}", 
                    template['operands']
                ),
                'rationale': self._generate_rationale(layer, template, variant)
            }
            extensions.append(extension)
        
        return extensions
    
    def _generate_opcode(self, base_opcode: str, suffix: str) -> str:
        """Generate specific opcode for instruction variant"""
        # Simple opcode generation - in practice, this would follow RISC-V encoding rules
        base_val = int(base_opcode, 16)
        suffix_hash = hash(suffix) % 16
        return f"0x{base_val + suffix_hash:02X}"
    
    def _estimate_speedup(self, layer: Dict[str, Any], category: str) -> float:
        """Estimate potential speedup from custom instruction"""
        base_speedup = {
            'neural_compute': 3.5,
            'activation': 2.0,
            'normalization': 2.5,
            'pooling': 2.2
        }
        
        speedup = base_speedup.get(category, 2.0)
        
        # Adjust based on layer characteristics
        if layer['percentage'] > 20:
            speedup *= 1.2
        if layer['flops_estimate'] > 1000000:
            speedup *= 1.1
        
        return round(speedup, 2)
    
    def _estimate_instruction_reduction(self, layer: Dict[str, Any], category: str) -> float:
        """Estimate instruction count reduction percentage"""
        base_reduction = {
            'neural_compute': 65.0,
            'activation': 40.0,
            'normalization': 50.0,
            'pooling': 45.0
        }
        
        reduction = base_reduction.get(category, 40.0)
        
        # Adjust based on complexity
        if layer['parameters'] > 100000:
            reduction *= 1.1
        
        return round(min(reduction, 80.0), 1)  # Cap at 80%
    
    def _generate_assembly_example(self, instruction_name: str, operands: List[str]) -> str:
        """Generate example assembly code for the instruction"""
        operand_examples = {
            'rs1': 'x10',  # Source register 1
            'rs2': 'x11',  # Source register 2
            'rs3': 'x12',  # Source register 3
            'rd': 'x13'    # Destination register
        }
        
        assembly_operands = [operand_examples.get(op, op) for op in operands]
        return f"{instruction_name.lower()} {', '.join(assembly_operands)}"
    
    def _generate_rationale(self, layer: Dict[str, Any], template: Dict[str, Any], 
                           variant: Dict[str, Any]) -> str:
        """Generate rationale for why this instruction is suggested"""
        rationale = f"Layer '{layer['name']}' ({layer['type']}) consumes {layer['percentage']:.1f}% "
        rationale += f"of total execution time. The proposed {template['base_name']}.{variant['suffix']} "
        rationale += f"instruction can accelerate {variant['desc']} operations by combining multiple "
        rationale += f"RISC-V instructions into a single specialized operation."
        
        if layer['flops_estimate'] > 1000000:
            rationale += f" With ~{layer['flops_estimate']:,} FLOPs, this layer would benefit significantly "
            rationale += "from hardware acceleration."
        
        return rationale
    
    def _deduplicate_extensions(self, extensions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate extensions"""
        seen = set()
        unique_extensions = []
        
        for ext in extensions:
            key = (ext['name'], ext['category'])
            if key not in seen:
                seen.add(key)
                unique_extensions.append(ext)
        
        return unique_extensions
    
    def _prioritize_extensions(self, extensions: List[Dict[str, Any]], 
                             profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sort extensions by priority/impact"""
        def priority_score(ext):
            score = 0
            score += ext['estimated_speedup'] * 10
            score += ext['instruction_reduction'] * 0.5
            
            # Bonus for high-impact categories
            category_bonus = {
                'neural_compute': 20,
                'activation': 10,
                'normalization': 15,
                'pooling': 12
            }
            score += category_bonus.get(ext['category'], 5)
            
            return score
        
        extensions.sort(key=priority_score, reverse=True)
        return extensions
    
    def export_to_assembly(self, extensions: List[Dict[str, Any]]) -> str:
        """Export extensions as pseudo-assembly code"""
        assembly_code = "# AI-Generated RISC-V ISA Extensions\n"
        assembly_code += "# Generated for neural network acceleration\n\n"
        
        for ext in extensions:
            assembly_code += f"# {ext['name']}: {ext['description']}\n"
            assembly_code += f"# Estimated speedup: {ext['estimated_speedup']}x\n"
            assembly_code += f"# Instruction reduction: {ext['instruction_reduction']}%\n"
            assembly_code += f"# Example usage: {ext['assembly_example']}\n"
            assembly_code += f".insn {ext['opcode']}\n\n"
        
        return assembly_code
    
    def export_to_json(self, extensions: List[Dict[str, Any]]) -> str:
        """Export extensions as JSON"""
        return json.dumps(extensions, indent=2, default=str)
