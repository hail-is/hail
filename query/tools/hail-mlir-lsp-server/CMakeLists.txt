set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(hail_dialect_libs GLOBAL PROPERTY HAIL_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
  ${mlir_dialect_libs}
  ${conversion_libs}
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  # Hail Libraries Go Here
  ${hail_dialect_libs}
  HailLowerToLLVM
  )

add_mlir_tool(hail-mlir-lsp-server
  hail-mlir-lsp-server.cpp

  DEPENDS
  ${LIBS}
  )
target_link_libraries(hail-mlir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(hail-mlir-lsp-server)

install(TARGETS hail-mlir-lsp-server
    RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
    COMPONENT hail-mlir-lsp-server)

mlir_check_all_link_libraries(hail-mlir-lsp-server)
