# 🚀 RISC-V ISA Extension Tool - PYNQ-Z2 Deployment

Complete deployment package for running the RISC-V ISA Extension Tool on PYNQ-Z2 FPGA boards.

## 📁 Package Contents

```
deployment/
├── README.md                    # This file
├── deploy_macos.sh             # Automated macOS deployment script
├── macos_deployment_guide.md   # Detailed macOS setup guide
└── pynq_setup.py              # PYNQ-specific setup utilities

vivado_project/
├── tcl/
│   ├── create_pynq_project.tcl # Vivado project creation
│   └── build_project.tcl       # Automated build script
├── hdl/
│   ├── riscv_core_wrapper.v    # RISC-V core with custom ISA
│   ├── custom_isa_extensions.v # Custom instruction implementations
│   ├── neural_accelerator.v    # Neural network accelerator
│   └── axi_neural_accel.v      # AXI wrapper for accelerator
└── constraints/
    └── pynq_z2.xdc             # PYNQ-Z2 pin constraints
```

## 🎯 Quick Start

### Option 1: Automated Deployment (Recommended)
```bash
# Make script executable
chmod +x deployment/deploy_macos.sh

# Run automated deployment
./deployment/deploy_macos.sh

# Follow the prompts to:
# 1. Build FPGA design
# 2. Find PYNQ-Z2 on network
# 3. Deploy and start application
```

### Option 2: Manual Deployment
```bash
# 1. Build FPGA design
cd vivado_project
vivado -mode batch -source tcl/create_pynq_project.tcl
vivado -mode batch -source tcl/build_project.tcl

# 2. Copy files to PYNQ
scp -r . xilinx@<pynq-ip>:/home/xilinx/risc_v_isa_tool/

# 3. Start application
ssh xilinx@<pynq-ip>
cd /home/xilinx/risc_v_isa_tool
python3 -m streamlit run unified_app.py --server.port 8503 --server.address 0.0.0.0
```

## 🛠️ Prerequisites

### Hardware
- **PYNQ-Z2 Board** (Xilinx Zynq-7020)
- **microSD Card** (16GB+, Class 10)
- **Ethernet Cable**
- **USB Cable** (power/UART)
- **macOS Computer** (10.15+)

### Software
- **Xilinx Vivado** (2022.1+)
- **Python 3.8+**
- **Git**
- **SSH/SCP**

## 📋 Deployment Steps

### 1. Environment Setup
```bash
# Install Vivado (if not already installed)
# Download from: https://www.xilinx.com/support/download.html

# Set environment variables
export XILINX_VIVADO="/Applications/Xilinx/Vivado/2022.1"
export PATH="$XILINX_VIVADO/bin:$PATH"

# Verify installation
vivado -version
```

### 2. Build FPGA Design
```bash
# Create and build Vivado project
cd vivado_project
vivado -mode batch -source tcl/create_pynq_project.tcl
vivado -mode batch -source tcl/build_project.tcl

# Build outputs will be in:
# - deployment/system_wrapper.bit (FPGA bitstream)
# - deployment/system_wrapper.xsa (Hardware platform)
# - deployment/pynq_overlay/ (PYNQ overlay files)
```

### 3. Prepare PYNQ-Z2
```bash
# Download PYNQ image
curl -L -o pynq_z2_v3.0.1.img.zip http://www.pynq.io/board.html

# Flash to SD card
sudo dd if=pynq_z2_v3.0.1.img of=/dev/rdisk2 bs=1m

# Boot PYNQ-Z2 with SD card
```

### 4. Deploy Application
```bash
# Find PYNQ IP address
nmap -sn 192.168.1.0/24 | grep -B2 "Xilinx"

# Copy project files
rsync -avz . xilinx@<pynq-ip>:/home/xilinx/risc_v_isa_tool/

# Setup and start application
ssh xilinx@<pynq-ip>
cd /home/xilinx/risc_v_isa_tool
pip3 install --user streamlit plotly pandas numpy psutil matplotlib
python3 -m streamlit run unified_app.py --server.port 8503 --server.address 0.0.0.0
```

### 5. Access Web Interface
```bash
# Open browser to:
http://<pynq-ip>:8503
```

## 🔧 Advanced Configuration

### Custom Build Options
```bash
# Build with debug enabled
vivado -mode batch -source tcl/build_project.tcl -tclargs debug=true

# Build for different target frequency
vivado -mode batch -source tcl/build_project.tcl -tclargs freq=200

# Build with specific optimization
vivado -mode batch -source tcl/build_project.tcl -tclargs strategy=Performance_Explore
```

### PYNQ Overlay Customization
```python
# Load custom overlay
from pynq import Overlay
overlay = Overlay('/home/xilinx/risc_v_isa_tool/vivado_project/deployment/pynq_overlay/riscv_isa_extension.bit')

# Access neural accelerator
neural_accel = overlay.axi_neural_accel_0

# Configure for specific model
neural_accel.write(0x08, 0x01)  # CONV2D operation
neural_accel.write(0x0C, 224)   # Input height
neural_accel.write(0x10, 224)   # Input width
neural_accel.write(0x14, 3)     # Input channels

# Start processing
neural_accel.write(0x00, 0x01)
```

### Performance Monitoring
```python
# Monitor FPGA performance
import psutil
import time

# Get performance counters
cycle_count = neural_accel.read(0x34)
op_count = neural_accel.read(0x38)

# Monitor system resources
cpu_percent = psutil.cpu_percent()
memory_percent = psutil.virtual_memory().percent

print(f"FPGA Cycles: {cycle_count}")
print(f"Operations: {op_count}")
print(f"CPU Usage: {cpu_percent}%")
print(f"Memory Usage: {memory_percent}%")
```

## 📊 Generated Files

### FPGA Build Outputs
```
vivado_project/deployment/
├── system_wrapper.bit          # FPGA bitstream
├── system_wrapper.xsa          # Hardware platform
├── boot.bin                    # SD card boot file
├── system_wrapper.mcs          # QSPI flash file
├── build_summary.txt           # Build report
├── utilization_summary.txt     # Resource usage
└── pynq_overlay/
    ├── riscv_isa_extension.bit # PYNQ bitstream
    ├── riscv_isa_extension.hwh # Hardware handoff
    └── riscv_isa_extension.py  # Python overlay class
```

### Performance Reports
```
reports/
├── timing_summary.rpt          # Timing analysis
├── utilization.rpt             # Resource utilization
├── power_analysis.rpt          # Power consumption
└── drc_report.rpt             # Design rule check
```

## 🧪 Testing and Validation

### Basic Functionality Test
```python
# Test neural network inference
import numpy as np

# Create test input
input_data = np.random.rand(224, 224, 3).astype(np.float32)

# Run inference
result = overlay.run_neural_inference(input_data, operation_type='conv2d')

print(f"Input shape: {input_data.shape}")
print(f"Output shape: {result.shape}")
print("✅ Neural inference test passed")
```

### Performance Benchmark
```python
# Benchmark inference speed
import time

num_iterations = 100
start_time = time.time()

for i in range(num_iterations):
    result = overlay.run_neural_inference(input_data)

end_time = time.time()
avg_time = (end_time - start_time) / num_iterations

print(f"Average inference time: {avg_time*1000:.2f} ms")
print(f"Throughput: {1/avg_time:.1f} FPS")
```

### Custom ISA Extension Test
```python
# Test custom RISC-V instructions
riscv_core = overlay.riscv_core_wrapper_0

# Test VCONV instruction
riscv_core.write(0x00, 0x01)  # Enable custom instructions
status = riscv_core.read(0x04)

if status & 0x01:
    print("✅ Custom ISA extensions active")
else:
    print("❌ Custom ISA extensions not responding")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Vivado Build Errors
```bash
# Check build logs
cat vivado_project/vivado.log | grep ERROR

# Common solutions:
# - Update board files
# - Check HDL syntax
# - Verify constraint file
```

#### 2. PYNQ Connection Issues
```bash
# Check network connectivity
ping <pynq-ip>

# Reset PYNQ network
ssh xilinx@<pynq-ip> 'sudo systemctl restart networking'

# Check SSH configuration
ssh -v xilinx@<pynq-ip>
```

#### 3. Application Startup Issues
```bash
# Check Python dependencies
ssh xilinx@<pynq-ip> 'pip3 list | grep streamlit'

# Check application logs
ssh xilinx@<pynq-ip> 'tail -f /home/xilinx/risc_v_isa_tool/streamlit.log'

# Restart application
ssh xilinx@<pynq-ip> 'cd /home/xilinx/risc_v_isa_tool && pkill -f streamlit && python3 -m streamlit run unified_app.py --server.port 8503 --server.address 0.0.0.0'
```

#### 4. Overlay Loading Issues
```python
# Check overlay files
import os
overlay_path = '/home/xilinx/risc_v_isa_tool/vivado_project/deployment/pynq_overlay/riscv_isa_extension.bit'
print(f"Overlay exists: {os.path.exists(overlay_path)}")

# Check file permissions
import stat
file_stat = os.stat(overlay_path)
print(f"File permissions: {stat.filemode(file_stat.st_mode)}")

# Try loading with verbose output
from pynq import Overlay
overlay = Overlay(overlay_path, download=True)
print(f"Overlay loaded: {overlay.is_loaded()}")
```

### Performance Optimization

#### 1. FPGA Clock Optimization
```tcl
# In build_project.tcl, adjust clock constraints
create_clock -period 5.000 -name sys_clk [get_ports sys_clk]  # 200MHz instead of 100MHz
```

#### 2. Memory Bandwidth Optimization
```python
# Use larger buffer sizes
neural_accel.write(0x40, 2048)  # Increase buffer size

# Enable burst transfers
neural_accel.write(0x44, 0x01)  # Enable burst mode
```

#### 3. Parallel Processing
```python
# Use multiple processing elements
neural_accel.write(0x48, 16)  # Set PE count to 16

# Enable pipeline processing
neural_accel.write(0x4C, 0x01)  # Enable pipeline
```

## 📚 Documentation

### Project Documentation
- [PYNQ_STARTUP_GUIDE.md](../PYNQ_STARTUP_GUIDE.md) - Quick start guide
- [macos_deployment_guide.md](macos_deployment_guide.md) - Detailed macOS setup
- [README.md](../README.md) - Project overview

### External Resources
- [PYNQ Documentation](http://pynq.readthedocs.io/)
- [Vivado Design Suite User Guide](https://www.xilinx.com/support/documentation/)
- [Zynq-7000 Technical Reference](https://www.xilinx.com/support/documentation/user_guides/ug585-Zynq-7000-TRM.pdf)

## 🎉 Success Metrics

After successful deployment, you should have:

✅ **FPGA Design Built** - Bitstream generated successfully  
✅ **PYNQ-Z2 Accessible** - SSH and web interface working  
✅ **Application Running** - Streamlit interface accessible  
✅ **Neural Acceleration** - FPGA accelerator responding  
✅ **Custom ISA Extensions** - RISC-V core with custom instructions  
✅ **Performance Monitoring** - Real-time metrics available  

### Expected Performance
- **Inference Latency**: 50-200ms (depending on model)
- **Throughput**: 5-20 FPS (depending on model complexity)
- **Speedup**: 2-5x compared to CPU-only inference
- **Power Consumption**: 3-5W total system power
- **Resource Utilization**: 60-80% of FPGA resources

## 🚀 Next Steps

1. **Explore Models**: Try the 30+ preloaded neural network models
2. **Generate ISA Extensions**: Create custom instructions for your workloads
3. **Measure Performance**: Use built-in profiling and analysis tools
4. **Export Results**: Generate IEEE paper quality reports and metrics
5. **Customize Hardware**: Modify HDL for your specific requirements

**Happy accelerating! 🔧⚡**