# Overview

The AI-Guided RISC-V ISA Extension Tool is an intelligent system that analyzes neural network performance and generates custom RISC-V instruction set extensions to accelerate embedded AI workloads. The project combines AI-powered profiling with instruction set architecture (ISA) extension generation to optimize neural network performance on RISC-V processors. It identifies computational bottlenecks in PyTorch and ONNX models, suggests custom RISC-V instructions, and provides performance simulation through a web-based emulator with real-time visualization.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Dashboard**: Web-based interface built with Streamlit for real-time visualization and analysis
- **Interactive UI**: Plotly-powered charts and graphs for performance metrics visualization
- **Web-based RISC-V Emulator**: HTML/JavaScript emulator for testing custom instructions in the browser

## Backend Architecture
- **Modular Python Architecture**: Organized into distinct modules for profiling, ISA generation, and analysis
- **Event-driven Profiling**: Uses PyTorch hooks to capture layer execution times and performance metrics
- **Template-based ISA Generation**: Rule-based system with predefined instruction templates for different neural network operations
- **Performance Analysis Engine**: Comparative analysis system for measuring improvements from ISA extensions

## Core Components
- **ModelProfiler**: Analyzes PyTorch/ONNX models using forward hooks to identify computational bottlenecks
- **ISAGenerator**: AI-powered generator that creates custom RISC-V instructions based on profiling data
- **PerformanceAnalyzer**: Compares baseline vs extended performance metrics and calculates improvements
- **Sample Model Generator**: Creates simplified MobileNet-like models for demonstration and testing

## Data Flow
1. Neural network models are loaded and profiled to identify performance bottlenecks
2. Profiling data is processed by the ISA generator to suggest custom RISC-V instructions
3. Performance analyzer compares baseline metrics with projected improvements
4. Results are visualized through the Streamlit dashboard and web emulator

## Design Patterns
- **Template Method Pattern**: ISA generation uses templates for different operation types (convolution, matrix multiplication, activation functions)
- **Observer Pattern**: Forward hooks implement observer pattern for layer execution monitoring
- **Strategy Pattern**: Different profiling strategies for PyTorch vs ONNX models
- **Factory Pattern**: Model creation and ISA instruction generation follow factory patterns

# External Dependencies

## Core ML/AI Libraries
- **PyTorch**: Neural network model handling, profiling, and forward hook registration
- **ONNX & ONNX Runtime**: Cross-platform neural network model format support and execution
- **TorchVision**: Pre-trained model access and computer vision utilities

## Web Framework & Visualization
- **Streamlit**: Web application framework for dashboard creation and real-time updates
- **Plotly**: Interactive charting and visualization library for performance metrics display
- **Pandas**: Data manipulation and analysis for profiling results processing

## Scientific Computing
- **NumPy**: Numerical computing foundation for performance calculations and data processing
- **JSON**: Configuration management and data serialization for ISA templates and analysis results

## Development Tools
- **Pathlib**: Modern path handling for cross-platform file operations
- **Subprocess**: System integration for external tool execution (potential QEMU integration)
- **Collections**: Advanced data structures for profiling data aggregation

## Browser-based Components
- **Web RISC-V Emulator**: Client-side JavaScript/WebAssembly emulator for instruction testing
- **HTML/CSS/JavaScript**: Frontend components for the web-based emulator interface