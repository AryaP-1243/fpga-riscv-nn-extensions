#!/bin/bash
# Compilation script for neural network with custom RISC-V ISA extensions

echo "Compiling neural network with custom ISA extensions..."

# Set RISC-V toolchain path (adjust as needed)
RISCV_TOOLCHAIN_PATH="/opt/riscv"
LLVM_PATH="/usr/bin"

# Compile flags
CFLAGS="-march=rv64gc -mabi=lp64d -O2 -std=c++17"
CUSTOM_FLAGS="-Illvm_backend -DCUSTOM_ISA_EXTENSIONS"

# Compile the neural network
${LLVM_PATH}/clang++ ${CFLAGS} ${CUSTOM_FLAGS} -o optimized_neural_network optimized_neural_network.cpp

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Output binary: optimized_neural_network"
    echo "To run: ./optimized_neural_network"
else
    echo "Compilation failed!"
    echo "Note: This requires RISC-V LLVM toolchain with custom ISA support"
    echo "For simulation, use QEMU or Spike RISC-V simulator"
fi
