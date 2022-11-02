#include "Dialect/Hail/IR/HailDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail;
using namespace hail::ir;

mlir::Operation *HailDialect::materializeConstant(mlir::OpBuilder &builder, mlir::Attribute value,
                                            mlir::Type type, mlir::Location loc) {
  assert(type.isa<IntegerType>());
  auto intAttr = value.cast<mlir::IntegerAttr>();
  assert(intAttr);
  auto i32Op = builder.create<I32Op>(loc, type, intAttr);
  return i32Op;
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/Hail/IR/HailOpsTypes.cpp.inc"
#include "Dialect/Hail/IR/HailOpsDialect.cpp.inc"

void HailDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Hail/IR/HailOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Hail/IR/HailOpsTypes.cpp.inc"
      >();
}
