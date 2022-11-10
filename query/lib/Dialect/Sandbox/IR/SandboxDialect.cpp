#include "Dialect/Sandbox/IR/Sandbox.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail;
using namespace hail::ir;

mlir::Operation *SandboxDialect::materializeConstant(mlir::OpBuilder &builder,
                                                     mlir::Attribute value,
                                                     mlir::Type type,
                                                     mlir::Location loc) {
  assert(type.isa<IntType>() || type.isa<BooleanType>());
  auto intAttr = value.cast<mlir::IntegerAttr>();
  assert(intAttr);
  auto constOp = builder.create<ConstantOp>(loc, type, intAttr);
  return constOp;
}

#define GET_TYPEDEF_CLASSES

#include "Dialect/Sandbox/IR/SandboxOpsDialect.cpp.inc"
#include "Dialect/Sandbox/IR/SandboxOpsTypes.cpp.inc"

void SandboxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Sandbox/IR/SandboxOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Sandbox/IR/SandboxOpsTypes.cpp.inc"
      >();
}
