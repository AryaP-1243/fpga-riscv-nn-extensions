//===-- RISCVCustomInstructions.cpp - Custom Instructions -----*- C++ -*-===//
//
// Custom RISC-V instruction implementation for neural network acceleration
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

namespace llvm {
namespace RISCV {

// Custom instruction implementations

// Implementation for VCONV.8 instruction
void emitVCONV.8(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                const RISCVInstrInfo *TII, unsigned DestReg,
                unsigned SrcReg1, unsigned SrcReg2) {
  BuildMI(MBB, MI, MI->getDebugLoc(), TII->get(RISCV::VCONV.8))
      .addReg(DestReg, RegState::Define)
      .addReg(SrcReg1)
      .addReg(SrcReg2);
}

// Implementation for RELU.V instruction
void emitRELU.V(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                const RISCVInstrInfo *TII, unsigned DestReg,
                unsigned SrcReg1, unsigned SrcReg2) {
  BuildMI(MBB, MI, MI->getDebugLoc(), TII->get(RISCV::RELU.V))
      .addReg(DestReg, RegState::Define)
      .addReg(SrcReg1)
      .addReg(SrcReg2);
}

} // namespace RISCV
} // namespace llvm
