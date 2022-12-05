set(LLVM_OPTIONAL_SOURCES
  MissingnessAnalysis.cpp
  MissingnessAwareConstantPropagationAnalysis.cpp
  )

add_mlir_library(HailAnalysis
  MissingnessAnalysis.cpp
  MissingnessAwareConstantPropagationAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Analysis

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  HailMissingDialect
  MLIRAnalysis
  # MLIRCallInterfaces
  MLIRControlFlowInterfaces
  # MLIRDataLayoutInterfaces
  # MLIRInferIntRangeInterface
  # MLIRInferTypeOpInterface
  # MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  # MLIRViewLikeInterface
  )
