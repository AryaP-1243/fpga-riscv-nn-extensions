// RISC-V Neural Network Extension Header
// Auto-generated for FPGA-accelerated RISC-V core
// Uses proper instruction encodings for custom ISA

#ifndef RISCV_NN_EXTENSIONS_H
#define RISCV_NN_EXTENSIONS_H

#include <stdint.h>

// Custom instruction opcodes
#define OP_CUSTOM0 0x0B
#define OP_CUSTOM1 0x2B

// Function codes
#define FUNCT7_CONV2D 0x00
#define FUNCT7_CONV2D_RELU 0x01
#define FUNCT7_POOL_MAX 0x02
#define FUNCT7_POOL_AVG 0x03
#define FUNCT7_POOL_MIN 0x04

// Inline assembly functions for custom instructions
// These use the exact encoding expected by your RISC-V core

static inline uint32_t fpga_vconv_raw(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    uint32_t instr = 0;
    // Encode: funct7=0x00, rs2, rs1, funct3=0x0, rd, opcode=0x0B
    instr |= (FUNCT7_CONV2D & 0x7F) << 25;
    instr |= (rs2 & 0x1F) << 20;
    instr |= (rs1 & 0x1F) << 15;
    instr |= (rd & 0x1F) << 7;
    instr |= (OP_CUSTOM0 & 0x7F);

    register uint32_t result __asm__("a0");
    __asm__ volatile (
        ".word %1\n\t"
        "mv %0, a0"
        : "=r"(result)
        : "i"(instr)
        : "a0"
    );
    return result;
}

static inline uint32_t fpga_vconv_relu_raw(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    uint32_t instr = 0;
    instr |= (FUNCT7_CONV2D_RELU & 0x7F) << 25;
    instr |= (rs2 & 0x1F) << 20;
    instr |= (rs1 & 0x1F) << 15;
    instr |= (rd & 0x1F) << 7;
    instr |= (OP_CUSTOM0 & 0x7F);

    register uint32_t result __asm__("a0");
    __asm__ volatile (
        ".word %1\n\t"
        "mv %0, a0"
        : "=r"(result)
        : "i"(instr)
        : "a0"
    );
    return result;
}

static inline uint32_t fpga_pool_max_raw(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    uint32_t instr = 0;
    instr |= (FUNCT7_POOL_MAX & 0x7F) << 25;
    instr |= (rs2 & 0x1F) << 20;
    instr |= (rs1 & 0x1F) << 15;
    instr |= (rd & 0x1F) << 7;
    instr |= (OP_CUSTOM0 & 0x7F);

    register uint32_t result __asm__("a0");
    __asm__ volatile (
        ".word %1\n\t"
        "mv %0, a0"
        : "=r"(result)
        : "i"(instr)
        : "a0"
    );
    return result;
}

static inline uint32_t fpga_pool_avg_raw(uint32_t rd, uint32_t rs1, uint32_t rs2) {
    uint32_t instr = 0;
    instr |= (FUNCT7_POOL_AVG & 0x7F) << 25;
    instr |= (rs2 & 0x1F) << 20;
    instr |= (rs1 & 0x1F) << 15;
    instr |= (rd & 0x1F) << 7;
    instr |= (OP_CUSTOM0 & 0x7F);

    register uint32_t result __asm__("a0");
    __asm__ volatile (
        ".word %1\n\t"
        "mv %0, a0"
        : "=r"(result)
        : "i"(instr)
        : "a0"
    );
    return result;
}

// High-level convenience functions
#define FPGA_VCONV(output_addr, input_addr, kernel_addr)     fpga_vconv_raw(output_addr, input_addr, kernel_addr)

#define FPGA_VCONV_RELU(output_addr, input_addr, kernel_addr)     fpga_vconv_relu_raw(output_addr, input_addr, kernel_addr)

#define FPGA_POOL_MAX(cycles_result, input_addr, output_addr)     fpga_pool_max_raw(cycles_result, input_addr, output_addr)

#define FPGA_POOL_AVG(cycles_result, input_addr, output_addr)     fpga_pool_avg_raw(cycles_result, input_addr, output_addr)

#endif // RISCV_NN_EXTENSIONS_H
