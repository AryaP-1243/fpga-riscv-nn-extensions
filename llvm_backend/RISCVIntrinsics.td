//===-- RISCVIntrinsics.td - RISC-V Custom Intrinsics --------*- tablegen -*-===//
//
// Custom RISC-V intrinsics for neural network acceleration
//
//===----------------------------------------------------------------------===//

// Base class for RISC-V custom intrinsics
class RISCVCustomIntrinsic<string name, list<LLVMType> ret_types,
                           list<LLVMType> param_types,
                           list<IntrinsicProperty> properties>
    : GCCBuiltin<"__builtin_riscv_" # name>,
      Intrinsic<ret_types, param_types, properties>;

def int_riscv_vconv.8 : RISCVCustomIntrinsic<"vconv.8",
    [llvm_void_ty],
    [llvm_ptr_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_relu.v : RISCVCustomIntrinsic<"relu.v",
    [llvm_void_ty],
    [llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty],
    [IntrNoMem, IntrHasSideEffects]>;
