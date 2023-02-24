add_hail_dialect_library(HailCPSDialect
        CPSDialect.cpp
        CPSOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/CPS/IR

        DEPENDS
        MLIRCPSOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
)
