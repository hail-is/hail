add_hail_dialect_library(HailOptionDialect
        OptionDialect.cpp
        OptionOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/Option/IR

        DEPENDS
        MLIROptionOpsIncGen

    LINK_LIBS PUBLIC
    HailCPSDialect
    MLIRIR
)
