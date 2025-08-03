//===-- RISCVInstrInfoCustom.td - Custom RISC-V Instructions -*- tablegen -*-===//
//
// Custom RISC-V instruction definitions for neural network acceleration
//
//===----------------------------------------------------------------------===//

// Custom instruction formats
class RISCVCustomInstruction<bits<7> opcode, bits<3> funct3, string opcodestr>
    : RISCVInstruction<(outs), (ins GPR:$rs1, GPR:$rs2, GPR:$rd),
                       opcodestr, "$rd, $rs1, $rs2", [], IIAlu> {
  bits<5> rs1;
  bits<5> rs2;
  bits<5> rd;
  
  let Inst{31-25} = 0b0000000;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Inst{6-0} = opcode;
}

// Vectorized 2D Convolution - 8-bit integer convolution
def VCONV.8 : RISCVCustomInstruction<0b1011011, 0b000, "vconv.8"> {
  let DecoderNamespace = "RISCV";
  let isAsCheapAsAMove = 0;
  let isMoveImm = 0;
}
// ReLU Activation Function - vectorized ReLU
def RELU.V : RISCVCustomInstruction<0b1011011, 0b001, "relu.v"> {
  let DecoderNamespace = "RISCV";
  let isAsCheapAsAMove = 0;
  let isMoveImm = 0;
}
