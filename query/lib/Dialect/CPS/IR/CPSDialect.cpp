#include "Dialect/CPS/IR/CPS.h"

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
#include "Dialect/CPS/IR/CPSOpsDialect.cpp.inc"
#include "Dialect/CPS/IR/CPSOpsTypes.cpp.inc"

void CPSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/CPS/IR/CPSOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/CPS/IR/CPSOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CallCCOp
//===----------------------------------------------------------------------===//

mlir::Type ContinuationType::parse(::mlir::AsmParser &parser) {
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
    parser.emitError(parser.getCurrentLocation(), "failed to parse CPS_ContType parameter 'inputs' which is to be a `::llvm::ArrayRef<mlir::Type>`");
    return {};
  }
  return ContinuationType::get(parser.getContext(), inputs);
}

void ContinuationType::print(::mlir::AsmPrinter &p) const {
  ::mlir::Builder odsBuilder(getContext());
  p << "<";
  p.printStrippedAttrOrType(getInputs());
  p << ">";
}

//===----------------------------------------------------------------------===//
// CallCCOp
//===----------------------------------------------------------------------===//

mlir::ParseResult CallCCOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the return continuation argument and return types
  mlir::OpAsmParser::UnresolvedOperand retContName;
  llvm::SmallVector<mlir::Type> retTypes;
  if (parser.parseOperand(retContName, /*allowResultNumber=*/false) ||
      parser.parseOptionalColonTypeList(retTypes))
    return mlir::failure();
  result.addTypes(retTypes);

  mlir::Type retContType = builder.getType<ContinuationType>(retTypes);
  mlir::OpAsmParser::Argument retContArg{retContName, retContType, {}, {}};

  // If attributes are present, parse them.
  mlir::NamedAttrList parsedAttributes;
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return mlir::failure();
  result.attributes.append(parsedAttributes);

  auto *body = result.addRegion();
  mlir::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*body, retContArg))
    return mlir::failure();
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return mlir::success();
}

void CallCCOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << getRegion().getArgument(0);
  if (getResultTypes().size() > 0) {
    p << " : " << getResultTypes();
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlocksArgs=*/false);
}

//===----------------------------------------------------------------------===//
// DefContOp
//===----------------------------------------------------------------------===//

mlir::ParseResult DefContOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the arguments list
  llvm::SmallVector<mlir::OpAsmParser::Argument> arguments;
  if (parser.parseArgumentList(arguments, mlir::OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes;
  argTypes.reserve(arguments.size());
  for (auto &arg : arguments)
    argTypes.push_back(arg.type);
  mlir::Type type = builder.getType<ContinuationType>(argTypes);
  result.addTypes(type);

  // If attributes are present, parse them.
  mlir::NamedAttrList parsedAttributes;
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return mlir::failure();
  result.attributes.append(parsedAttributes);

  // Parse the body
  auto *body = result.addRegion();
  mlir::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(*body, arguments))
    return mlir::failure();
  if (body->empty())
    return parser.emitError(loc, "expected non-empty function body");

  return mlir::success();
}

void DefContOp::print(mlir::OpAsmPrinter &p) {
  // Print the arguments list
  p << '(';
  llvm::interleaveComma(getRegion().getArguments(), p, [&](auto arg){
    p.printRegionArgument(arg);
  });
  p << ')';

  // If attributes are present, print them.
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  // Print the body
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlocksArgs=*/false);
}

//===----------------------------------------------------------------------===//
// ApplyContOp
//===----------------------------------------------------------------------===//

mlir::ParseResult ApplyContOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  auto &builder = parser.getBuilder();

  // Parse the continuation and argument names
  mlir::OpAsmParser::UnresolvedOperand contName;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> argNames;

  if (parser.parseOperand(contName) ||
      parser.parseOperandList(argNames, mlir::OpAsmParser::Delimiter::Paren))
    return mlir::failure();

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Value, 1> cont;
  llvm::SmallVector<mlir::Value> args;
  mlir::NamedAttrList parsedAttributes;
  result.attributes.append(parsedAttributes);
  argTypes.reserve(argNames.size());
  args.reserve(argNames.size());
  if (parser.parseOptionalColonTypeList(argTypes) ||
      parser.resolveOperand(contName, builder.getType<ContinuationType>(argTypes), cont) ||
      parser.resolveOperands(argNames, argTypes, parser.getCurrentLocation(), args) ||
      parser.parseOptionalAttrDict(parsedAttributes))
    return mlir::failure();

  result.addOperands(cont);
  result.addOperands(args);
  result.attributes.append(parsedAttributes);
  return mlir::success();
}

void ApplyContOp::print(mlir::OpAsmPrinter &p) {
  p << ' ' << cont() << '(' << args() << ')';
  if (args().size() > 0) {
    p << " : " << args().getTypes();
  }
  p.printOptionalAttrDict(getOperation()->getAttrs());
}
