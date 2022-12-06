# Declare the library associated with a dialect.
function(add_hail_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY HAIL_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction()
