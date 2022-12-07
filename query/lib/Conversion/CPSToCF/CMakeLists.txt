add_mlir_conversion_library(HailCPSToCF
  CPSToCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Conversion/CPSToCF

  DEPENDS
  HailConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  HailCPSDialect
  MLIRControlFlowDialect
  MLIRPass
  MLIRTransforms
  )
