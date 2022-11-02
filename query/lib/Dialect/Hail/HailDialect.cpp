#include "Dialect/Hail/HailDialect.h"
#include "Dialect/Hail/HailOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail;
using namespace hail::ir;

#define GET_TYPEDEF_CLASSES
#include "Dialect/Hail/HailOpsTypes.cpp.inc"
#include "Dialect/Hail/HailOpsDialect.cpp.inc"

void HailDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Hail/HailOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Hail/HailOpsTypes.cpp.inc"
      >();
}
