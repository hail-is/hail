add_hail_dialect_library(HailSandboxDialect
        SandboxDialect.cpp
        SandboxOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Sandbox/IR

        DEPENDS
        MLIRSandboxOpsIncGen
        MLIRSandboxOpsAttributesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
