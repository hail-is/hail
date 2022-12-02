#include "Dialect/Sandbox/IR/Sandbox.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail;
using namespace hail::ir;

auto SandboxDialect::materializeConstant(mlir::OpBuilder &builder, mlir::Attribute value,
                                         mlir::Type type, mlir::Location loc) -> mlir::Operation * {
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

void ir::ArrayType::walkImmediateSubElements(
    mlir::function_ref<void(mlir::Attribute)> walkAttrsFn,
    mlir::function_ref<void(mlir::Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

auto ir::ArrayType::replaceImmediateSubElements(mlir::ArrayRef<mlir::Attribute> replAttrs,
                                                mlir::ArrayRef<Type> replTypes) const
    -> mlir::Type {
  return get(getContext(), replTypes.front());
}

auto ir::ArrayType::verify(mlir::function_ref<mlir::InFlightDiagnostic()> emitError,
                           Type elementType) -> mlir::LogicalResult {
  if (elementType.isa<ir::BooleanType>() || elementType.isa<ir::IntType>() ||
      elementType.isa<ir::ArrayType>()) {
    return mlir::success();
  }

  return mlir::failure();
}

auto ir::ArrayType::parse(::mlir::AsmParser &parser) -> mlir::Type {
  mlir::Type elementType;
  if (parser.parseLess())
    return {};
  if (parser.parseType(elementType))
    return nullptr;
  if (parser.parseGreater())
    return {};
  return ir::ArrayType::get(parser.getContext(), elementType);
}

void ir::ArrayType::print(::mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << '<' << getElementType() << '>';
}