add_mlir_dialect_library(MLIRHail
        HailDialect.cpp
        HailOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Hail/IR

        DEPENDS
        MLIRHailOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
