#include "hail/Dialect/Sandbox/IR/Sandbox.h"

#include "hail/Support/MLIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail::ir;

auto SandboxDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                         Location loc) -> Operation * {
  assert(type.isa<IntType>() || type.isa<BooleanType>());
  auto intAttr = value.cast<IntegerAttr>();
  assert(intAttr);
  auto constOp = builder.create<ConstantOp>(loc, type, intAttr);
  return constOp;
}

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/Sandbox/IR/SandboxOpsDialect.cpp.inc"
#include "hail/Dialect/Sandbox/IR/SandboxOpsTypes.cpp.inc"

void SandboxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hail/Dialect/Sandbox/IR/SandboxOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hail/Dialect/Sandbox/IR/SandboxOpsTypes.cpp.inc"
      >();
}

void ArrayType::walkImmediateSubElements(llvm::function_ref<void(Attribute)> walkAttrsFn,
                                         llvm::function_ref<void(Type)> walkTypesFn) const {
  walkTypesFn(getElementType());
}

auto ArrayType::replaceImmediateSubElements(ArrayRef<Attribute> replAttrs,
                                            ArrayRef<Type> replTypes) const -> Type {
  return get(getContext(), replTypes.front());
}

auto ArrayType::verify(llvm::function_ref<InFlightDiagnostic()> emitError, Type elementType)
    -> LogicalResult {
  if (elementType.isa<BooleanType>() || elementType.isa<IntType>()
      || elementType.isa<ArrayType>()) {
    return success();
  }

  return failure();
}

auto ArrayType::parse(mlir::AsmParser &parser) -> Type {
  Type elementType;
  if (parser.parseLess())
    return {};
  if (parser.parseType(elementType))
    return nullptr;
  if (parser.parseGreater())
    return {};
  return ArrayType::get(parser.getContext(), elementType);
}

void ArrayType::print(mlir::AsmPrinter &odsPrinter) const {
  odsPrinter << '<' << getElementType() << '>';
}
