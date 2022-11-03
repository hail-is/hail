add_mlir_dialect_library(MLIRSandbox
        SandboxDialect.cpp
        SandboxOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Sandbox/IR

        DEPENDS
        MLIRSandboxOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
