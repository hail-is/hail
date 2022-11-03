set(LLVM_TARGET_DEFINITIONS SandboxOps.td)
mlir_tablegen(SandboxOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SandboxOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(SandboxOps sb)