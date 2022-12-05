add_mlir_conversion_library(HailOptionToGenericOption
  OptionToGenericOption.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Conversion/OptionToGenericOption

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HailCPSDialect
  HailOptionDialect
  MLIRPass
  MLIRTransforms
  )
