set(LLVM_TARGET_DEFINITIONS SandboxOps.td)
mlir_tablegen(SandboxOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(SandboxOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(SandboxOps sb)

mlir_tablegen(SandboxOpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SandboxOpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSandboxOpsAttributesIncGen)
