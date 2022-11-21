get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(hail_dialect_libs GLOBAL PROPERTY HAIL_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${mlir_dialect_libs}
        ${conversion_libs}
        MLIROptLib
        # Hail Libraries Go Here
        ${hail_dialect_libs}
        HailTransforms
        )
add_llvm_executable(hail-opt hail-opt.cpp)
install(TARGETS hail-opt
    RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
    COMPONENT hail-opt)

llvm_update_compile_flags(hail-opt)
target_link_libraries(hail-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hail-opt)
