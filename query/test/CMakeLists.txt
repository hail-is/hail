configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

find_program(LLVM_LIT NAMES "llvm-lit" "lit"
    HINTS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_BUILD_BINARY_DIR})

if(LLVM_LIT)
    message(STATUS "Found llvm-lit: ${LLVM_LIT}")
    get_filename_component(LLVM_TEST_BINARY_DIR "${LLVM_LIT}" DIRECTORY)

    find_program(FILE_CHECK "FileCheck"
                 HINTS ${LLVM_TOOLS_BINARY_DIR}
                       ${LLVM_TEST_BINARY_DIR}
                       ${LLVM_BUILD_BINARY_DIR}
                 REQUIRED)
    message(STATUS "Found FileCheck: ${FILE_CHECK}")

    add_custom_target(check COMMAND "${LLVM_LIT}" "${CMAKE_CURRENT_BINARY_DIR}" -v
        DEPENDS hail-opt)
else()
    message(WARNING "llvm-lit not found, disabling tests")
endif()
