add_mlir_conversion_library(HailLowerOption
  LowerOption.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Conversion/LowerOption

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HailCPSDialect
  HailOptionDialect
  MLIRArithmeticDialect
  MLIRPass
  MLIRTransforms
  )
