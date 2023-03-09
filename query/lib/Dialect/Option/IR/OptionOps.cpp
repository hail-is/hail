#include "hail/Dialect/Option/IR/Option.h"

#include "hail/Support/MLIR.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "hail/Dialect/Option/IR/OptionOps.cpp.inc"

using namespace hail::ir;

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

void MapOp::build(OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultValueTypes,
                  ValueRange inputs) {
  odsState.addTypes(odsBuilder.getType<OptionType>(resultValueTypes));
  auto *region = odsState.addRegion();
  region->emplaceBlock();
  llvm::SmallVector<Type> argTypes;
  for (auto input : inputs) {
    auto valueTypes = input.getType().cast<OptionType>().getValueTypes();
    argTypes.append(valueTypes.begin(), valueTypes.end());
  }
  llvm::SmallVector<Location> const locs(argTypes.size(), odsState.location);
  region->addArguments(argTypes, locs);
}

//===----------------------------------------------------------------------===//
// ConstructOp
//===----------------------------------------------------------------------===//

void ConstructOp::build(OpBuilder &odsBuilder, OperationState &odsState, TypeRange valueTypes) {
  odsState.addTypes(odsBuilder.getType<OptionType>(valueTypes));
  auto *region = odsState.addRegion();
  region->emplaceBlock();
  region->addArgument(odsBuilder.getType<ContinuationType>(), odsState.location);
  region->addArgument(odsBuilder.getType<ContinuationType>(valueTypes), odsState.location);
}

//===----------------------------------------------------------------------===//
// DestructOp
//===----------------------------------------------------------------------===//

namespace {

struct DestructOfConstruct : public OpRewritePattern<DestructOp> {
  using OpRewritePattern<DestructOp>::OpRewritePattern;

  auto matchAndRewrite(DestructOp destruct, PatternRewriter &rewriter) const
      -> LogicalResult override {
    ConstructOp source;
    size_t sourceValuesStart = 0;
    size_t curValueIdx = 0;
    llvm::SmallVector<Type> sourceValueTypes;
    llvm::SmallVector<Type> remainingValueTypes;
    llvm::SmallVector<Value> remainingOptions;
    for (auto const &input : llvm::enumerate(destruct.inputs())) {
      if (auto construct = input.value().getDefiningOp<ConstructOp>()) {
        if (!source && construct->hasOneUse()) {
          source = construct;
          auto valueTypes = construct.getType().getValueTypes();
          sourceValuesStart = curValueIdx;
          curValueIdx += valueTypes.size();
          sourceValueTypes.append(valueTypes.begin(), valueTypes.end());
        } else {
          auto valueTypes = input.value().getType().cast<OptionType>().getValueTypes();
          curValueIdx += valueTypes.size();
          remainingValueTypes.append(valueTypes.begin(), valueTypes.end());
          remainingOptions.push_back(input.value());
        }
      }
    }

    if (!source)
      return failure();

    // Define present continuation to pass to source. It will destruct all remaining options, with a
    // present continuation that merges its values with those of this continuation, and pases them
    // on to the original present continuation.
    auto cont = rewriter.create<DefContOp>(destruct.getLoc(), sourceValueTypes);

    rewriter.mergeBlockBefore(&source.bodyRegion().front(), destruct,
                              {destruct.missingCont(), cont});

    // Define present continuation for the new destruct op, taking all remaining values, after
    // removing those of 'source'.
    rewriter.setInsertionPointToStart(cont.getBody());
    auto cont2 = rewriter.create<DefContOp>(destruct.getLoc(), remainingValueTypes);

    // Create new destruct op taking all remaining options.
    rewriter.create<DestructOp>(destruct.getLoc(), remainingOptions, destruct.missingCont(), cont2);

    // Define body of 'cont2' to forward all values on to original present continuation.
    rewriter.setInsertionPointToStart(cont2.getBody());
    llvm::SmallVector<Value> newValues(cont2.bodyRegion().args_begin(),
                                       cont2.bodyRegion().args_end());
    newValues.insert(newValues.begin() + sourceValuesStart, cont.bodyRegion().args_begin(),
                     cont.bodyRegion().args_end());
    rewriter.create<ApplyContOp>(destruct.getLoc(), destruct.presentCont(), newValues);

    rewriter.eraseOp(destruct);
    rewriter.eraseOp(source);
    return success();
  }
};

struct EmptyDestruct : public OpRewritePattern<DestructOp> {
  using OpRewritePattern<DestructOp>::OpRewritePattern;

  auto matchAndRewrite(DestructOp destruct, PatternRewriter &rewriter) const
      -> LogicalResult override {
    if (!destruct.inputs().empty())
      return failure();

    rewriter.replaceOpWithNewOp<ApplyContOp>(destruct, destruct.presentCont(),
                                             llvm::ArrayRef<Value>{});
    return success();
  }
};

} // end namespace

void DestructOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  results.add<DestructOfConstruct, EmptyDestruct>(context);
}

auto DestructOp::parse(mlir::OpAsmParser &parser, OperationState &result) -> mlir::ParseResult {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputNames;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> contNames;
  llvm::SmallVector<Type> optTypes;
  NamedAttrList parsedAttributes;
  if (parser.parseOperandList(inputNames, mlir::OpAsmParser::Delimiter::Paren)
      || parser.parseOperandList(contNames, 2, mlir::OpAsmParser::Delimiter::Square)
      || parser.parseOptionalAttrDict(parsedAttributes)
      || parser.parseOptionalColonTypeList(optTypes))
    return failure();

  llvm::SmallVector<Value> inputs;
  inputs.reserve(inputNames.size());
  if (parser.resolveOperands(inputNames, optTypes, parser.getCurrentLocation(), inputs))
    return failure();

  llvm::SmallVector<Type> valueTypes;
  for (auto t : optTypes) {
    auto optType = t.dyn_cast<OptionType>();
    if (!optType)
      return failure();
    valueTypes.append(optType.getValueTypes().begin(), optType.getValueTypes().end());
  }
  llvm::SmallVector<Value> conts;
  if (parser.resolveOperands(
          contNames,
          {builder.getType<ContinuationType>(), builder.getType<ContinuationType>(valueTypes)},
          parser.getCurrentLocation(), conts))
    return failure();

  result.addOperands(inputs);
  result.addOperands(conts);
  result.addAttributes(parsedAttributes);

  return success();
}

void DestructOp::print(mlir::OpAsmPrinter &p) {
  p << '(';
  p.printOperands(inputs());
  p << ')';
  p << '[';
  p.printOperand(missingCont());
  p << ", ";
  p.printOperand(presentCont());
  p << ']';

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  p << ' ';
  if (!inputs().empty()) {
    p << " : " << inputs().getTypes();
  }
}
