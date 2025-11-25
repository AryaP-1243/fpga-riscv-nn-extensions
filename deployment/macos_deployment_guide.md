# 🍎 macOS Deployment Guide for RISC-V ISA Extension Tool on PYNQ-Z2

Complete guide for deploying the RISC-V ISA Extension Tool on PYNQ-Z2 using macOS with Vivado.

## 📋 Prerequisites

### 1. Hardware Requirements
- **PYNQ-Z2 Board** (Xilinx Zynq-7020 FPGA)
- **microSD Card** (16GB+ recommended, Class 10)
- **USB-A to Micro-USB Cable** (for power and UART)
- **Ethernet Cable** (for network connectivity)
- **macOS Computer** (macOS 10.15+ recommended)

### 2. Software Requirements
- **Xilinx Vivado** (2022.1 or later)
- **Python 3.8+** with pip
- **Git** (for version control)
- **Terminal** or **iTerm2**
- **SD Card Formatter** (for SD card preparation)

## 🛠️ Step 1: Install Vivado on macOS

### Download and Install Vivado
1. **Download Vivado**:
   ```bash
   # Visit Xilinx website and download Vivado ML Edition
   # https://www.xilinx.com/support/download.html
   ```

2. **Install Vivado**:
   ```bash
   # Mount the downloaded installer
   # Run the installer and select:
   # - Vivado ML Edition
   # - Include Zynq-7000 devices
   # - Include PYNQ-Z2 board files
   ```

3. **Set Environment Variables**:
   ```bash
   # Add to ~/.zshrc or ~/.bash_profile
   export XILINX_VIVADO="/Applications/Xilinx/Vivado/2022.1"
   export PATH="$XILINX_VIVADO/bin:$PATH"
   
   # Reload shell configuration
   source ~/.zshrc
   ```

4. **Verify Installation**:
   ```bash
   vivado -version
   # Should show Vivado version information
   ```

### Install PYNQ Board Files
```bash
# Download PYNQ board files
git clone https://github.com/Xilinx/XilinxBoardStore.git
cd XilinxBoardStore

# Copy board files to Vivado installation
sudo cp -r boards/* $XILINX_VIVADO/data/boards/board_files/

# Verify PYNQ-Z2 board files
ls $XILINX_VIVADO/data/boards/board_files/ | grep pynq
```

## 🔧 Step 2: Setup Development Environment

### 1. Clone the Project
```bash
# Clone the RISC-V ISA Extension Tool
git clone <your-repository-url>
cd RISC-V_Extension

# Verify project structure
ls -la
# Should see: vivado_project/, deployment/, *.py files, etc.
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install streamlit plotly pandas numpy psutil matplotlib seaborn
pip install jupyter notebook ipywidgets

# For PYNQ development (optional)
pip install pynq-utils
```

### 3. Setup RISC-V Toolchain (Optional)
```bash
# Install RISC-V GNU toolchain for cross-compilation
brew install riscv-gnu-toolchain

# Verify installation
riscv64-unknown-elf-gcc --version
```

## 🏗️ Step 3: Build FPGA Design

### 1. Create Vivado Project
```bash
# Navigate to project directory
cd vivado_project

# Launch Vivado in batch mode
vivado -mode batch -source tcl/create_pynq_project.tcl

# Or launch Vivado GUI
vivado &
# Then: Tools -> Run Tcl Script -> select create_pynq_project.tcl
```

### 2. Build the Design
```bash
# Run complete build process
vivado -mode batch -source tcl/build_project.tcl

# This will:
# - Run synthesis
# - Run implementation  
# - Generate bitstream
# - Create deployment files
```

### 3. Monitor Build Progress
```bash
# Check build logs
tail -f vivado_project/vivado.log

# Build typically takes 30-60 minutes depending on your Mac
```

### 4. Verify Build Results
```bash
# Check deployment directory
ls -la vivado_project/deployment/
# Should contain:
# - system_wrapper.bit
# - system_wrapper.xsa
# - boot.bin
# - build_summary.txt

# Check PYNQ overlay files
ls -la vivado_project/deployment/pynq_overlay/
# Should contain:
# - riscv_isa_extension.bit
# - riscv_isa_extension.hwh
# - riscv_isa_extension.py
```

## 💾 Step 4: Prepare PYNQ-Z2 SD Card

### 1. Download PYNQ Image
```bash
# Download latest PYNQ image for Z2
curl -L -o pynq_z2_v3.0.1.img.zip \
  http://www.pynq.io/board.html

# Extract image
unzip pynq_z2_v3.0.1.img.zip
```

### 2. Flash SD Card
```bash
# Find SD card device
diskutil list
# Look for your SD card (e.g., /dev/disk2)

# Unmount SD card
diskutil unmountDisk /dev/disk2

# Flash PYNQ image (replace disk2 with your SD card)
sudo dd if=pynq_z2_v3.0.1.img of=/dev/rdisk2 bs=1m
# This takes 10-20 minutes

# Eject SD card
diskutil eject /dev/disk2
```

### 3. Configure PYNQ
```bash
# Re-insert SD card and mount it
# Edit boot partition files if needed

# Copy our overlay files to SD card
cp vivado_project/deployment/pynq_overlay/* /Volumes/BOOT/overlays/
```

## 🌐 Step 5: Deploy to PYNQ-Z2

### 1. Boot PYNQ-Z2
1. Insert SD card into PYNQ-Z2
2. Connect Ethernet cable
3. Connect USB cable for power
4. Set boot jumper to SD card mode
5. Power on the board

### 2. Find PYNQ IP Address
```bash
# Check your router's DHCP table, or use:
nmap -sn 192.168.1.0/24 | grep -B2 "Xilinx"

# Or connect via USB UART
screen /dev/tty.usbserial-* 115200
# Login: xilinx / xilinx
# Check IP: ifconfig eth0
```

### 3. SSH to PYNQ
```bash
# SSH to PYNQ board
ssh xilinx@<pynq-ip-address>
# Password: xilinx

# Update system
sudo apt update && sudo apt upgrade -y

# Install additional packages
sudo apt install -y htop git vim
```

### 4. Transfer Project Files
```bash
# From your Mac, copy files to PYNQ
scp -r . xilinx@<pynq-ip>:/home/xilinx/risc_v_isa_tool/

# Or use rsync for faster transfer
rsync -avz --progress . xilinx@<pynq-ip>:/home/xilinx/risc_v_isa_tool/
```

## 🚀 Step 6: Run the Application

### 1. Setup Python Environment on PYNQ
```bash
# SSH to PYNQ
ssh xilinx@<pynq-ip>

# Navigate to project
cd /home/xilinx/risc_v_isa_tool

# Install Python dependencies
pip3 install --user streamlit plotly pandas numpy psutil matplotlib

# Run deployment script
python3 pynq_deployment_guide.py
```

### 2. Start Web Interface
```bash
# Start Streamlit application
python3 -m streamlit run unified_app.py --server.port 8503 --server.address 0.0.0.0

# Or use the PYNQ-optimized version
python3 pynq_optimized_app.py
```

### 3. Access Web Interface
```bash
# Open browser on your Mac and navigate to:
http://<pynq-ip>:8503

# You should see the RISC-V ISA Extension Tool interface
```

## 🧪 Step 7: Test and Validate

### 1. Load FPGA Overlay
```python
# In Jupyter notebook or Python shell on PYNQ
from pynq import Overlay
import numpy as np

# Load our custom overlay
overlay = Overlay('/home/xilinx/risc_v_isa_tool/vivado_project/deployment/pynq_overlay/riscv_isa_extension.bit')

# Check loaded IP blocks
print(overlay.ip_dict.keys())
```

### 2. Test Neural Accelerator
```python
# Test neural network acceleration
input_data = np.random.rand(224, 224, 3).astype(np.float32)

# Configure accelerator
neural_accel = overlay.axi_neural_accel_0
neural_accel.write(0x08, 0x01)  # Set operation type to CONV2D
neural_accel.write(0x0C, 224)   # Input height
neural_accel.write(0x10, 224)   # Input width
neural_accel.write(0x14, 3)     # Input channels

# Start processing
neural_accel.write(0x00, 0x01)  # Start bit

# Check status
status = neural_accel.read(0x04)
print(f"Accelerator status: {status}")
```

### 3. Test Custom ISA Extensions
```python
# Test RISC-V core with custom instructions
riscv_core = overlay.riscv_core_wrapper_0

# Read version register
version = riscv_core.read(0x3C)
print(f"RISC-V core version: 0x{version:08x}")

# Test custom instruction execution
# (Implementation depends on your specific RISC-V core)
```

### 4. Run Performance Analysis
```bash
# Run the complete analysis suite
python3 ieee_performance_analysis.py

# This generates:
# - Performance comparison plots
# - Resource utilization analysis
# - IEEE paper quality metrics
```

## 📊 Step 8: Monitor and Debug

### 1. System Monitoring
```bash
# Monitor system resources
htop

# Check FPGA temperature
cat /sys/class/hwmon/hwmon0/temp1_input

# Monitor network traffic
iftop -i eth0
```

### 2. Debug Tools
```bash
# Check kernel messages
dmesg | tail -20

# Monitor system logs
journalctl -f

# Check PYNQ overlay status
cat /proc/device-tree/chosen/overlays/*/status
```

### 3. Performance Profiling
```python
# Profile neural network inference
import time
import psutil

# Measure inference time
start_time = time.time()
# ... run inference ...
end_time = time.time()

print(f"Inference time: {(end_time - start_time)*1000:.2f} ms")
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Vivado Build Errors
```bash
# Check Vivado log files
cat vivado_project/vivado.log | grep ERROR

# Common fixes:
# - Update board files
# - Check constraint file syntax
# - Verify HDL syntax
```

#### 2. PYNQ Boot Issues
```bash
# Check SD card integrity
fsck /dev/disk2s1

# Re-flash SD card if corrupted
# Check boot jumper settings
```

#### 3. Network Connectivity
```bash
# Check Ethernet connection
ping <pynq-ip>

# Reset network on PYNQ
sudo systemctl restart networking

# Check firewall settings
sudo ufw status
```

#### 4. Overlay Loading Issues
```python
# Check overlay file integrity
import os
os.path.exists('/path/to/overlay.bit')

# Verify hardware handoff file
os.path.exists('/path/to/overlay.hwh')

# Check PYNQ version compatibility
import pynq
print(pynq.__version__)
```

### Performance Optimization Tips

1. **Use Ethernet for file transfers** (faster than USB)
2. **Enable swap on PYNQ** for large models
3. **Use high-speed SD card** (Class 10 or better)
4. **Monitor FPGA temperature** during intensive operations
5. **Optimize Vivado settings** for your specific use case

## 📚 Additional Resources

### Documentation
- [PYNQ Documentation](http://pynq.readthedocs.io/)
- [Vivado Design Suite User Guide](https://www.xilinx.com/support/documentation/sw_manuals/xilinx2022_1/ug973-vivado-release-notes-install-license.pdf)
- [Zynq-7000 Technical Reference Manual](https://www.xilinx.com/support/documentation/user_guides/ug585-Zynq-7000-TRM.pdf)

### Community Support
- [PYNQ Community Forum](http://www.pynq.io/community.html)
- [Xilinx Forums](https://forums.xilinx.com/)
- [RISC-V Foundation](https://riscv.org/)

### Example Projects
- [PYNQ Neural Network Examples](https://github.com/Xilinx/PYNQ-ComputerVision)
- [RISC-V on FPGA Examples](https://github.com/cliffordwolf/picorv32)

## 🎉 Success!

You now have a complete RISC-V ISA Extension Tool running on PYNQ-Z2! 

### What You Can Do:
- ✅ Analyze neural network models
- ✅ Generate custom RISC-V instructions
- ✅ Accelerate inference on FPGA
- ✅ Generate IEEE paper quality metrics
- ✅ Export results for publication

### Next Steps:
1. **Experiment with different models** using the 30+ preloaded options
2. **Generate custom ISA extensions** for your specific workloads
3. **Measure performance improvements** with the built-in profiling tools
4. **Export results** for academic papers or commercial presentations

**Happy accelerating! 🚀**