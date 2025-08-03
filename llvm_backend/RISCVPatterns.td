//===-- RISCVPatterns.td - Custom RISC-V Patterns --------*- tablegen -*-===//
//
// Custom RISC-V pattern matching for neural network optimizations
//
//===----------------------------------------------------------------------===//

// Pattern for VCONV.8 instruction
def : Pat<(int_riscv_vconv.8 GPR:$rs1, GPR:$rs2, GPR:$rd),
          (VCONV.8 GPR:$rs1, GPR:$rs2, GPR:$rd)>;
// Pattern for RELU.V instruction
def : Pat<(int_riscv_relu.v GPR:$rs1, GPR:$rs2, GPR:$rd),
          (RELU.V GPR:$rs1, GPR:$rs2, GPR:$rd)>;
