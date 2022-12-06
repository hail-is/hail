//===------------------------------------------------------------------------------------------===//
//
// Modification of mlir/Support/LLVM.h
// This file forward declares and imports various common LLVM and MLIR datatypes that Hail wants to
// use unqualified.
//
// Note that most of these are forward declared and then imported into the MLIR namespace with using
// decls, rather than being #included.  This is because we want clients to explicitly #include the
// files they need.
//
//===------------------------------------------------------------------------------------------===//

#ifndef HAIL_SUPPORT_MLIR_H
#define HAIL_SUPPORT_MLIR_H

// We include these two headers because they cannot be practically forward
// declared, and are effectively language features.
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Casting.h"
#include <vector>

// Forward declarations.
namespace llvm {

// String types
template <unsigned N>
class SmallString;
class StringRef;
class StringLiteral;
class Twine;

// Containers.
template <typename T>
class ArrayRef;
class BitVector;
namespace detail {
template <typename KeyT, typename ValueT>
struct DenseMapPair;
} // namespace detail
template <typename KeyT, typename ValueT, typename KeyInfoT, typename BucketT>
class DenseMap;
template <typename T, typename Enable>
struct DenseMapInfo;
template <typename ValueT, typename ValueInfoT>
class DenseSet;
class MallocAllocator;
template <typename T>
class MutableArrayRef;
template <typename T>
class Optional;
template <typename... PT>
class PointerUnion;
template <typename T, typename Vector, typename Set>
class SetVector;
template <typename T, unsigned N>
class SmallPtrSet;
template <typename T>
class SmallPtrSetImpl;
template <typename T, unsigned N>
class SmallVector;
template <typename T>
class SmallVectorImpl;
template <typename AllocatorTy>
class StringSet;
template <typename T, typename R>
class StringSwitch;
template <typename T>
class TinyPtrVector;
template <typename T, typename ResultT>
class TypeSwitch;

// Other common classes.
class APInt;
class APSInt;
class APFloat;
template <typename Fn>
class function_ref;
template <typename IteratorT>
class iterator_range;
class raw_ostream;
class SMLoc;
class SMRange;

} // namespace llvm

namespace mlir {

// Core IR types
class Attribute;
class Block;
class Location;
class Operation;
class Type;
class TypeRange;
class Value;
class ValueRange;

// Types and Attributes
class ArrayAttr;
class BoolAttr;
class DictionaryAttr;
class FlatSymbolRefAttr;
class IntegerAttr;
class IntegerType;
class NamedAttrList;
class RankedTensorType;
class SymbolRefAttr;

// Common classes
class Builder;
class ConversionPatternRewriter;
class ConversionTarget;
class IRRewriter;
class InFlightDiagnostic;
struct LogicalResult;
class MLIRContext;
class OpBuilder;
template <typename SourceOp>
class OpConversionPattern;
struct OperationState;
class OpFoldResult;
template <typename SourceOp>
struct OpRewritePattern;
class Pass;
class PatternRewriter;
class RewritePatternSet;

// Common functions
// NOLINTBEGIN(readability-redundant-declaration)
auto success(bool isSuccess) -> LogicalResult;
auto failure(bool isFailure) -> LogicalResult;
// NOLINTEND(readability-redundant-declaration)

} // namespace mlir

namespace hail::ir {
// NOLINTBEGIN(misc-unused-using-decls)

// Casting operators.
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;
using llvm::isa_and_nonnull;

// String types
using llvm::SmallString;
using llvm::StringLiteral;
using llvm::StringRef;
using llvm::Twine;

// Container Related types
//
// Containers.
using llvm::ArrayRef;
using llvm::BitVector;
using llvm::DenseMap;
using llvm::DenseMapInfo;
using llvm::DenseSet;
using llvm::MutableArrayRef;
using llvm::None;
using llvm::Optional;
using llvm::PointerUnion;
using llvm::SetVector;
using llvm::SmallPtrSet;
using llvm::SmallPtrSetImpl;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringSet;
using llvm::StringSwitch;
using llvm::TinyPtrVector;
using llvm::TypeSwitch;

// Other common classes.
using llvm::APFloat;
using llvm::APInt;
using llvm::APSInt;
using llvm::function_ref;
using llvm::iterator_range;
using llvm::raw_ostream;
using llvm::SMLoc;
using llvm::SMRange;

// Core MLIR IR classes
using mlir::Attribute;
using mlir::Block;
using mlir::Location;
using mlir::Operation;
using mlir::Type;
using mlir::TypeRange;
using mlir::Value;
using mlir::ValueRange;

// MLIR Types and Attributes
using mlir::ArrayAttr;
using mlir::BoolAttr;
using mlir::DictionaryAttr;
using mlir::FlatSymbolRefAttr;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::NamedAttrList;
using mlir::RankedTensorType;
using mlir::SymbolRefAttr;

// Common MLIR classes
using mlir::Builder;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::InFlightDiagnostic;
using mlir::IRRewriter;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::OpRewritePattern;
using mlir::Pass;
using mlir::PatternRewriter;
using mlir::RewritePatternSet;

// Common functions
using mlir::failure;
using mlir::success;

// NOLINTEND(misc-unused-using-decls)
} // namespace hail::ir

#endif // HAIL_SUPPORT_MLIR_H
