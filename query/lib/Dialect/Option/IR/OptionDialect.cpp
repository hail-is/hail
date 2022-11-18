#include "Dialect/Option/IR/Option.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail::ir;

#define GET_TYPEDEF_CLASSES
#include "Dialect/Option/IR/OptionOpsDialect.cpp.inc"
#include "Dialect/Option/IR/OptionOpsTypes.cpp.inc"

void OptionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Option/IR/OptionOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Option/IR/OptionOpsTypes.cpp.inc"
      >();
}
