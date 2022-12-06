#include "hail/Dialect/Missing/IR/Missing.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail::ir;

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/Missing/IR/MissingOpsDialect.cpp.inc"

void MissingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hail/Dialect/Missing/IR/MissingOps.cpp.inc"
      >();
}
