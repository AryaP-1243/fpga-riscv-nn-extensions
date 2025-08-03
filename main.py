#!/usr/bin/env python3
"""
AI-Guided Instruction Set Extension for RISC-V
Main orchestrator for the neural network profiling and ISA generation pipeline
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project directories to path
sys.path.append(str(Path(__file__).parent))

from profiler.torch_profiler import ModelProfiler
from isa_engine.isa_generator import ISAGenerator
from utils.analysis_tools import PerformanceAnalyzer
import streamlit as st

def run_profiling_pipeline(model_path=None, model_type="mobilenet"):
    """
    Run the complete profiling and ISA generation pipeline
    
    Args:
        model_path: Path to model file (optional)
        model_type: Type of model to use if no path provided
    
    Returns:
        dict: Results containing profiling data, ISA suggestions, and analysis
    """
    results = {}
    
    try:
        # Step 1: Initialize profiler
        print("Initializing model profiler...")
        profiler = ModelProfiler()
        
        # Step 2: Load and profile model
        if model_path and os.path.exists(model_path):
            profile_data = profiler.profile_model_from_file(model_path)
        else:
            profile_data = profiler.profile_sample_model(model_type)
        
        results['profiling'] = profile_data
        print(f"Profiling complete. Found {len(profile_data['layers'])} layers.")
        
        # Step 3: Generate ISA extensions
        print("Generating ISA extensions...")
        isa_generator = ISAGenerator()
        isa_extensions = isa_generator.generate_extensions(profile_data)
        results['isa_extensions'] = isa_extensions
        print(f"Generated {len(isa_extensions)} ISA extension suggestions.")
        
        # Step 4: Analyze performance improvements
        print("Analyzing performance improvements...")
        analyzer = PerformanceAnalyzer()
        analysis = analyzer.analyze_improvements(profile_data, isa_extensions)
        results['analysis'] = analysis
        
        print("Pipeline completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        results['error'] = str(e)
        return results

def save_results(results, output_file="results.json"):
    """Save pipeline results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="AI-Guided RISC-V ISA Extension Tool")
    parser.add_argument("--model", help="Path to model file (ONNX format)")
    parser.add_argument("--model-type", default="mobilenet", 
                       choices=["mobilenet", "resnet", "custom"],
                       help="Type of model to use")
    parser.add_argument("--output", default="results.json", 
                       help="Output file for results")
    parser.add_argument("--dashboard", action="store_true",
                       help="Launch Streamlit dashboard")
    
    args = parser.parse_args()
    
    if args.dashboard:
        print("Launching Streamlit dashboard...")
        os.system("streamlit run dashboard/app.py --server.port 5000")
    else:
        # Run CLI pipeline
        print("Starting AI-Guided RISC-V ISA Extension Pipeline...")
        results = run_profiling_pipeline(args.model, args.model_type)
        save_results(results, args.output)

if __name__ == "__main__":
    main()
