add_mlir_dialect_library(MLIRHail
        Hail/HailDialect.cpp
        Hail/HailOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Hail

        DEPENDS
        MLIRHailOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
