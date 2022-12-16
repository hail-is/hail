add_hail_dialect_library(HailMissingDialect
        MissingDialect.cpp
        MissingOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Missing/IR

        DEPENDS
        MLIRMissingOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
