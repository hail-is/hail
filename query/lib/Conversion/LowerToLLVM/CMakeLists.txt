add_mlir_conversion_library(HailLowerToLLVM
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Conversion/LowerToLLVM

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HailSandboxDialect
  MLIRControlFlowToLLVM
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  )
