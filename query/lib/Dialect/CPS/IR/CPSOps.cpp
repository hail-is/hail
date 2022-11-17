#include "Dialect/CPS/IR/CPS.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define GET_OP_CLASSES

#include "Dialect/CPS/IR/CPSOps.cpp.inc"
namespace hail {
namespace ir {

} // namespace ir
} // namespace hail
