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

namespace hail::ir::detail {

struct OptionTypeStorage final
    : public mlir::TypeStorage,
      public llvm::TrailingObjects<OptionTypeStorage, mlir::Type> {
  using KeyTy = mlir::TypeRange;

  OptionTypeStorage(unsigned numTypes) : numElements(numTypes) {}

  /// Construction.
  static OptionTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      mlir::TypeRange key) {
    // Allocate a new storage instance.
    auto byteSize = OptionTypeStorage::totalSizeToAlloc<mlir::Type>(key.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OptionTypeStorage));
    auto *result = ::new (rawMem) OptionTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(),
                            result->getTrailingObjects<mlir::Type>());
    return result;
  }

  bool operator==(const KeyTy &key) const { return key == getTypes(); }

  /// Return the number of held types.
  unsigned size() const { return numElements; }

  /// Return the held types.
  llvm::ArrayRef<mlir::Type> getTypes() const {
    return {getTrailingObjects<mlir::Type>(), size()};
  }

  /// The number of tuple elements.
  unsigned numElements;
};

} // namespace hail::ir::detail

llvm::ArrayRef<mlir::Type> OptionType::getValueTypes() const { return getImpl()->getTypes(); }

mlir::Type OptionType::parse(::mlir::AsmParser &parser) {
  mlir::Builder builder(parser.getContext());
  llvm::SmallVector<mlir::Type> inputs;

  auto parseType = [&](){
    auto element = mlir::FieldParser<mlir::Type>::parse(parser);
    if (failed(element))
      return mlir::failure();
    inputs.push_back(*element);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, parseType)) {
    parser.emitError(parser.getCurrentLocation(), "failed to parse OptionType parameter 'valueTypes' which is to be a `::llvm::ArrayRef<mlir::Type>`");
    return {};
  }
  return OptionType::get(parser.getContext(), inputs);
}

void OptionType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getValueTypes());
  odsPrinter << ">";
}
