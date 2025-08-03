"""
Performance Analysis Tools
Utilities for analyzing and comparing performance improvements from ISA extensions
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import json

class PerformanceAnalyzer:
    """Analyze performance improvements from custom ISA extensions"""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.extended_metrics = {}
        self.comparison_results = {}
    
    def analyze_improvements(self, profile_data: Dict[str, Any], 
                           isa_extensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze performance improvements from ISA extensions
        
        Args:
            profile_data: Profiling results from ModelProfiler
            isa_extensions: List of ISA extensions from ISAGenerator
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Calculate baseline metrics
            baseline = self._calculate_baseline_metrics(profile_data)
            
            # Calculate extended metrics
            extended = self._calculate_extended_metrics(profile_data, isa_extensions)
            
            # Compare and analyze
            analysis = self._compare_metrics(baseline, extended)
            
            # Add detailed breakdown
            analysis['breakdown'] = self._create_detailed_breakdown(
                profile_data, isa_extensions, baseline, extended
            )
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing improvements: {str(e)}")
    
    def _calculate_baseline_metrics(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate baseline performance metrics"""
        total_time = profile_data['total_time']
        layers = profile_data['layers']
        
        # Estimate baseline instruction count (simplified model)
        total_instructions = 0
        total_cycles = 0
        total_energy = 0  # Arbitrary units
        
        for layer in layers:
            # Rough instruction count estimation based on layer type and parameters
            layer_instructions = self._estimate_layer_instructions(layer)
            layer_cycles = layer_instructions * 1.2  # Assume some pipeline stalls
            layer_energy = layer_instructions * 0.1  # Rough energy per instruction
            
            total_instructions += layer_instructions
            total_cycles += layer_cycles
            total_energy += layer_energy
        
        return {
            'execution_time': total_time,
            'instruction_count': total_instructions,
            'cycle_count': total_cycles,
            'energy_consumption': total_energy,
            'code_size': total_instructions * 4  # Assume 4 bytes per instruction
        }
    
    def _calculate_extended_metrics(self, profile_data: Dict[str, Any], 
                                  isa_extensions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics with ISA extensions applied"""
        baseline = self._calculate_baseline_metrics(profile_data)
        
        # Apply improvements from ISA extensions
        total_speedup = 1.0
        total_instruction_reduction = 0.0
        affected_time_percentage = 0.0
        
        # Group extensions by target layer
        extensions_by_layer = {}
        for ext in isa_extensions:
            layer_name = ext['target_layer']
            if layer_name not in extensions_by_layer:
                extensions_by_layer[layer_name] = []
            extensions_by_layer[layer_name].append(ext)
        
        # Calculate impact for each affected layer
        for layer in profile_data['layers']:
            if layer['name'] in extensions_by_layer:
                layer_extensions = extensions_by_layer[layer['name']]
                
                # Take the best extension for this layer (highest speedup)
                best_extension = max(layer_extensions, key=lambda x: x['estimated_speedup'])
                
                layer_time_fraction = layer['percentage'] / 100.0
                layer_speedup = best_extension['estimated_speedup']
                
                # Apply Amdahl's law for this layer
                affected_time_percentage += layer_time_fraction
                
                # Weighted average of instruction reduction
                layer_reduction = best_extension['instruction_reduction']
                total_instruction_reduction += layer_reduction * layer_time_fraction
        
        # Calculate overall speedup using Amdahl's law
        if affected_time_percentage > 0:
            # Average speedup for affected portions
            avg_speedup = sum(ext['estimated_speedup'] for ext in isa_extensions) / len(isa_extensions)
            total_speedup = 1 / ((1 - affected_time_percentage) + (affected_time_percentage / avg_speedup))
        
        # Calculate new metrics
        new_execution_time = baseline['execution_time'] / total_speedup
        new_instruction_count = baseline['instruction_count'] * (1 - total_instruction_reduction / 100)
        new_cycle_count = baseline['cycle_count'] / total_speedup
        new_energy = baseline['energy_consumption'] * 0.8  # Assume 20% energy reduction
        new_code_size = baseline['code_size'] * 1.02  # Slight increase due to custom instructions
        
        return {
            'execution_time': new_execution_time,
            'instruction_count': new_instruction_count,
            'cycle_count': new_cycle_count,
            'energy_consumption': new_energy,
            'code_size': new_code_size,
            'speedup_factor': total_speedup,
            'instruction_reduction_percent': total_instruction_reduction
        }
    
    def _estimate_layer_instructions(self, layer: Dict[str, Any]) -> int:
        """Estimate number of instructions for a layer"""
        layer_type = layer['type'].lower()
        parameters = layer.get('parameters', 0)
        flops = layer.get('flops_estimate', 0)
        
        if 'conv' in layer_type:
            # Convolution layers are instruction-heavy
            return max(flops // 10, parameters * 5)
        elif 'linear' in layer_type or 'gemm' in layer_type:
            # Matrix multiplication
            return max(flops // 5, parameters * 2)
        elif 'relu' in layer_type:
            # Simple activation
            return max(flops // 100, 100)
        elif 'batchnorm' in layer_type:
            # Batch normalization
            return max(flops // 20, parameters * 10)
        elif 'pool' in layer_type:
            # Pooling operations
            return max(flops // 50, 500)
        else:
            # Default estimation
            return max(parameters, flops // 50, 100)
    
    def _compare_metrics(self, baseline: Dict[str, Any], 
                        extended: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline and extended metrics"""
        return {
            'overall_speedup': baseline['execution_time'] / extended['execution_time'],
            'cycle_reduction': (baseline['cycle_count'] - extended['cycle_count']) / baseline['cycle_count'] * 100,
            'total_instruction_reduction': (baseline['instruction_count'] - extended['instruction_count']) / baseline['instruction_count'] * 100,
            'estimated_energy_savings': (baseline['energy_consumption'] - extended['energy_consumption']) / baseline['energy_consumption'] * 100,
            'code_size_change': (extended['code_size'] - baseline['code_size']) / baseline['code_size'] * 100,
            'baseline_metrics': baseline,
            'extended_metrics': extended
        }
    
    def _create_detailed_breakdown(self, profile_data: Dict[str, Any], 
                                 isa_extensions: List[Dict[str, Any]],
                                 baseline: Dict[str, Any], 
                                 extended: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed breakdown of improvements"""
        breakdown = {
            'layer_improvements': [],
            'instruction_category_impact': {},
            'critical_path_analysis': {}
        }
        
        # Analyze improvements per layer
        extensions_by_layer = {}
        for ext in isa_extensions:
            layer_name = ext['target_layer']
            if layer_name not in extensions_by_layer:
                extensions_by_layer[layer_name] = []
            extensions_by_layer[layer_name].append(ext)
        
        for layer in profile_data['layers']:
            if layer['name'] in extensions_by_layer:
                layer_extensions = extensions_by_layer[layer['name']]
                best_extension = max(layer_extensions, key=lambda x: x['estimated_speedup'])
                
                improvement = {
                    'layer_name': layer['name'],
                    'layer_type': layer['type'],
                    'original_time_percent': layer['percentage'],
                    'estimated_speedup': best_extension['estimated_speedup'],
                    'instruction_reduction': best_extension['instruction_reduction'],
                    'isa_extension_used': best_extension['name'],
                    'impact_score': layer['percentage'] * best_extension['estimated_speedup']
                }
                breakdown['layer_improvements'].append(improvement)
        
        # Sort by impact score
        breakdown['layer_improvements'].sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Analyze by instruction category
        category_impact = {}
        for ext in isa_extensions:
            category = ext['category']
            if category not in category_impact:
                category_impact[category] = {
                    'total_speedup': 0,
                    'total_reduction': 0,
                    'instruction_count': 0
                }
            category_impact[category]['total_speedup'] += ext['estimated_speedup']
            category_impact[category]['total_reduction'] += ext['instruction_reduction']
            category_impact[category]['instruction_count'] += 1
        
        # Calculate averages
        for category, data in category_impact.items():
            count = data['instruction_count']
            data['avg_speedup'] = data['total_speedup'] / count
            data['avg_reduction'] = data['total_reduction'] / count
        
        breakdown['instruction_category_impact'] = category_impact
        
        return breakdown
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a text report of the analysis"""
        report = "RISC-V ISA Extension Performance Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall metrics
        report += "OVERALL PERFORMANCE IMPROVEMENTS:\n"
        report += f"• Overall Speedup: {analysis['overall_speedup']:.2f}x\n"
        report += f"• Instruction Count Reduction: {analysis['total_instruction_reduction']:.1f}%\n"
        report += f"• Estimated Energy Savings: {analysis['estimated_energy_savings']:.1f}%\n"
        report += f"• Code Size Change: {analysis['code_size_change']:.1f}%\n\n"
        
        # Baseline vs Extended
        baseline = analysis['baseline_metrics']
        extended = analysis['extended_metrics']
        
        report += "DETAILED METRICS COMPARISON:\n"
        report += f"                        Baseline    Extended    Improvement\n"
        report += f"Execution Time (s):     {baseline['execution_time']:.4f}    {extended['execution_time']:.4f}    {analysis['overall_speedup']:.2f}x\n"
        report += f"Instruction Count:      {baseline['instruction_count']:8d}  {extended['instruction_count']:8.0f}  {analysis['total_instruction_reduction']:.1f}%\n"
        report += f"Cycle Count:           {baseline['cycle_count']:8.0f}  {extended['cycle_count']:8.0f}  {analysis['cycle_reduction']:.1f}%\n"
        report += f"Energy (arb. units):   {baseline['energy_consumption']:8.1f}  {extended['energy_consumption']:8.1f}  {analysis['estimated_energy_savings']:.1f}%\n\n"
        
        # Layer improvements
        if 'breakdown' in analysis and 'layer_improvements' in analysis['breakdown']:
            report += "TOP LAYER IMPROVEMENTS:\n"
            for i, improvement in enumerate(analysis['breakdown']['layer_improvements'][:5]):
                report += f"{i+1}. {improvement['layer_name']} ({improvement['layer_type']}):\n"
                report += f"   • Original time: {improvement['original_time_percent']:.1f}% of total\n"
                report += f"   • Speedup: {improvement['estimated_speedup']:.1f}x\n"
                report += f"   • Instruction reduction: {improvement['instruction_reduction']:.1f}%\n"
                report += f"   • ISA extension: {improvement['isa_extension_used']}\n\n"
        
        # Category analysis
        if 'breakdown' in analysis and 'instruction_category_impact' in analysis['breakdown']:
            report += "INSTRUCTION CATEGORY ANALYSIS:\n"
            for category, data in analysis['breakdown']['instruction_category_impact'].items():
                report += f"• {category.title()}:\n"
                report += f"  - Average speedup: {data['avg_speedup']:.1f}x\n"
                report += f"  - Average instruction reduction: {data['avg_reduction']:.1f}%\n"
                report += f"  - Number of extensions: {data['instruction_count']}\n\n"
        
        return report
    
    def export_analysis(self, analysis: Dict[str, Any], format: str = 'json') -> str:
        """Export analysis results in specified format"""
        if format.lower() == 'json':
            return json.dumps(analysis, indent=2, default=str)
        elif format.lower() == 'text':
            return self.generate_report(analysis)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def calculate_roi(self, analysis: Dict[str, Any], 
                     development_cost: float = 100000,
                     production_volume: int = 10000) -> Dict[str, Any]:
        """
        Calculate Return on Investment for ISA extensions
        
        Args:
            analysis: Performance analysis results
            development_cost: Cost to develop the ISA extensions
            production_volume: Expected production volume
            
        Returns:
            ROI analysis
        """
        # Estimate benefits
        performance_gain = analysis['overall_speedup'] - 1  # Percentage gain
        energy_savings = analysis['estimated_energy_savings'] / 100
        
        # Assume benefits translate to cost savings
        performance_value = performance_gain * 50 * production_volume  # $50 per unit per x improvement
        energy_value = energy_savings * 20 * production_volume  # $20 per unit per % energy saving
        
        total_benefits = performance_value + energy_value
        net_benefit = total_benefits - development_cost
        roi_percentage = (net_benefit / development_cost) * 100 if development_cost > 0 else 0
        
        return {
            'development_cost': development_cost,
            'production_volume': production_volume,
            'performance_value': performance_value,
            'energy_value': energy_value,
            'total_benefits': total_benefits,
            'net_benefit': net_benefit,
            'roi_percentage': roi_percentage,
            'payback_period_units': development_cost / (total_benefits / production_volume) if total_benefits > 0 else float('inf')
        }
