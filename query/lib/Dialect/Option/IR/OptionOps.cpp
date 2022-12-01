#include "Dialect/Option/IR/Option.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

#define GET_OP_CLASSES
#include "Dialect/Option/IR/OptionOps.cpp.inc"

using namespace hail::ir;

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

void MapOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
                  mlir::TypeRange resultValueTypes, mlir::ValueRange inputs) {
  odsState.addTypes(odsBuilder.getType<OptionType>(resultValueTypes));
  auto region = odsState.addRegion();
  region->emplaceBlock();
  llvm::SmallVector<mlir::Type> argTypes;
  for (auto input : inputs) {
    auto valueTypes = input.getType().cast<OptionType>().getValueTypes();
    argTypes.append(valueTypes.begin(), valueTypes.end());
  }
  llvm::SmallVector<mlir::Location> locs(argTypes.size(), odsState.location);
  region->addArguments(argTypes, locs);
}

//===----------------------------------------------------------------------===//
// ConstructOp
//===----------------------------------------------------------------------===//

void ConstructOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
                        mlir::TypeRange valueTypes) {
  odsState.addTypes(odsBuilder.getType<OptionType>(valueTypes));
  auto region = odsState.addRegion();
  region->emplaceBlock();
  region->addArgument(odsBuilder.getType<ContinuationType>(), odsState.location);
  region->addArgument(odsBuilder.getType<ContinuationType>(valueTypes), odsState.location);
}

//===----------------------------------------------------------------------===//
// DestructOp
//===----------------------------------------------------------------------===//

namespace {

struct DestructOfConstruct : public mlir::OpRewritePattern<DestructOp> {
  using OpRewritePattern<DestructOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(DestructOp destruct,
                                      mlir::PatternRewriter &rewriter) const override {
    ConstructOp source;
    int sourceValuesStart;
    int curValueIdx = 0;
    llvm::SmallVector<mlir::Type> sourceValueTypes;
    llvm::SmallVector<mlir::Type> remainingValueTypes;
    llvm::SmallVector<mlir::Value> remainingOptions;
    for (const auto &input : llvm::enumerate(destruct.inputs())) {
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
      return mlir::failure();

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
    llvm::SmallVector<mlir::Value> newValues(cont2.bodyRegion().args_begin(),
                                             cont2.bodyRegion().args_end());
    newValues.insert(newValues.begin() + sourceValuesStart, cont.bodyRegion().args_begin(),
                     cont.bodyRegion().args_end());
    rewriter.create<ApplyContOp>(destruct.getLoc(), destruct.presentCont(), newValues);

    rewriter.eraseOp(destruct);
    rewriter.eraseOp(source);
    return mlir::success();
  }
};

struct EmptyDestruct : public mlir::OpRewritePattern<DestructOp> {
  using OpRewritePattern<DestructOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(DestructOp destruct,
                                      mlir::PatternRewriter &rewriter) const override {
    if (!destruct.inputs().empty())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<ApplyContOp>(destruct, destruct.presentCont(),
                                             llvm::ArrayRef<mlir::Value>{});
    return mlir::success();
  }
};

} // end namespace

void DestructOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             mlir::MLIRContext *context) {
  results.add<DestructOfConstruct, EmptyDestruct>(context);
}

mlir::ParseResult DestructOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputNames;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> contNames;
  llvm::SmallVector<mlir::Type> optTypes;
  mlir::NamedAttrList parsedAttributes;
  if (parser.parseOperandList(inputNames, mlir::OpAsmParser::Delimiter::Paren) ||
      parser.parseOperandList(contNames, 2, mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDict(parsedAttributes) || parser.parseOptionalColonTypeList(optTypes))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> inputs;
  inputs.reserve(inputNames.size());
  if (parser.resolveOperands(inputNames, optTypes, parser.getCurrentLocation(), inputs))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> valueTypes;
  for (auto t : optTypes) {
    auto optType = t.dyn_cast<OptionType>();
    if (!optType)
      return mlir::failure();
    valueTypes.append(optType.getValueTypes().begin(), optType.getValueTypes().end());
  }
  llvm::SmallVector<mlir::Value> conts;
  if (parser.resolveOperands(
          contNames,
          {builder.getType<ContinuationType>(), builder.getType<ContinuationType>(valueTypes)},
          parser.getCurrentLocation(), conts))
    return mlir::failure();

  result.addOperands(inputs);
  result.addOperands(conts);
  result.addAttributes(parsedAttributes);

  return mlir::success();
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
