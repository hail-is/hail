add_mlir_conversion_library(HailLowerSandbox
  LowerSandbox.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Conversion/LowerSandbox

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HailSandboxDialect
  MLIRArithmeticDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  )
