add_mlir_library(HailTransforms
  TestMissingnessAnalysis.cpp 

  ADDITIONAL_HEADER_DIRS
  ${HAIL_MAIN_INCLUDE_DIR}/Transforms

  DEPENDS
  HailTransformsPassIncGen

  LINK_LIBS PUBLIC
  HailAnalysis
  MLIRAnalysis
  MLIRCopyOpInterface
  MLIRLoopLikeInterface
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  )
