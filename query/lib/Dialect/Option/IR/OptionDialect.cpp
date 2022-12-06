#include "hail/Dialect/Option/IR/Option.h"

#include "hail/Support/MLIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace hail::ir;

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/Option/IR/OptionOpsDialect.cpp.inc"
#include "hail/Dialect/Option/IR/OptionOpsTypes.cpp.inc"

void OptionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hail/Dialect/Option/IR/OptionOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "hail/Dialect/Option/IR/OptionOpsTypes.cpp.inc"
      >();
}

namespace hail::ir::detail {

struct OptionTypeStorage final : public mlir::TypeStorage,
                                 public llvm::TrailingObjects<OptionTypeStorage, Type> {
  using KeyTy = TypeRange;

  OptionTypeStorage(unsigned numTypes) : numElements(numTypes) {}

  /// Construction.
  static auto construct(mlir::TypeStorageAllocator &allocator, TypeRange key)
      -> OptionTypeStorage * {
    // Allocate a new storage instance.
    auto byteSize = OptionTypeStorage::totalSizeToAlloc<Type>(key.size());
    auto *rawMem = allocator.allocate(byteSize, alignof(OptionTypeStorage));
    // NOLINTNEXTLINE(*-owning-memory)
    auto *result = ::new (rawMem) OptionTypeStorage(key.size());

    // Copy in the element types into the trailing storage.
    std::uninitialized_copy(key.begin(), key.end(), result->getTrailingObjects<Type>());
    return result;
  }

  auto operator==(KeyTy const &key) const -> bool { return key == getTypes(); }

  /// Return the number of held types.
  auto size() const -> unsigned { return numElements; }

  /// Return the held types.
  auto getTypes() const -> ArrayRef<Type> { return {getTrailingObjects<Type>(), size()}; }

  /// The number of tuple elements.
  unsigned numElements;
};

} // namespace hail::ir::detail

auto OptionType::getValueTypes() const -> ArrayRef<Type> { return getImpl()->getTypes(); }

auto OptionType::parse(mlir::AsmParser &parser) -> Type {
  Builder const builder(parser.getContext());
  SmallVector<Type> inputs;

  auto parseType = [&]() {
    auto element = mlir::FieldParser<Type>::parse(parser);
    if (failed(element))
      return failure();
    inputs.push_back(*element);
    return success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::LessGreater, parseType)) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse OptionType parameter 'valueTypes' which is to be a "
                     "`ArrayRef<Type>`");
    return {};
  }
  return OptionType::get(parser.getContext(), inputs);
}

void OptionType::print(mlir::AsmPrinter &odsPrinter) const {
  Builder const odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getValueTypes());
  odsPrinter << ">";
}
