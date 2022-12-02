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

struct OptionTypeStorage final : public mlir::TypeStorage,
                                 public llvm::TrailingObjects<OptionTypeStorage, mlir::Type> {
  using KeyTy = mlir::TypeRange;

  OptionTypeStorage(unsigned numTypes) : numElements(numTypes) {}

  /// Construction.
  static auto construct(mlir::TypeStorageAllocator &allocator, mlir::TypeRange key)
      -> OptionTypeStorage * {
    // Allocate a new storage instance.
    auto byteSize = OptionTypeStorage::totalSizeToAlloc<mlir::Type>(key.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OptionTypeStorage));
    // NOLINTNEXTLINE(*-owning-memory)
    auto *result = ::new (rawMem) OptionTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(), result->getTrailingObjects<mlir::Type>());
    return result;
  }

  auto operator==(KeyTy const &key) const -> bool { return key == getTypes(); }

  /// Return the number of held types.
  auto size() const -> unsigned { return numElements; }

  /// Return the held types.
  auto getTypes() const -> llvm::ArrayRef<mlir::Type> {
    return {getTrailingObjects<mlir::Type>(), size()};
  }

  /// The number of tuple elements.
  unsigned numElements;
};

} // namespace hail::ir::detail

auto OptionType::getValueTypes() const -> llvm::ArrayRef<mlir::Type> {
  return getImpl()->getTypes();
}

auto OptionType::parse(::mlir::AsmParser &parser) -> mlir::Type {
  mlir::Builder builder(parser.getContext());
  llvm::SmallVector<mlir::Type> inputs;

  auto parseType = [&]() {
    auto element = mlir::FieldParser<mlir::Type>::parse(parser);
    if (failed(element))
      return mlir::failure();
    inputs.push_back(*element);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, parseType)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse OptionType parameter 'valueTypes' which is to be a "
                     "`::llvm::ArrayRef<mlir::Type>`");
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
