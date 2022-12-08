#ifndef HAIL_INITALLDIALECTS_H_
#define HAIL_INITALLDIALECTS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/CPS/IR/CPS.h"
#include "Dialect/Missing/IR/Missing.h"
#include "Dialect/Option/IR/Option.h"
#include "Dialect/Sandbox/IR/Sandbox.h"

namespace hail::ir {

inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<mlir::arith::ArithDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect,
                  hail::ir::CPSDialect,
                  hail::ir::MissingDialect,
                  hail::ir::OptionDialect,
                  hail::ir::SandboxDialect>();
  // clang-format on
}

inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace hail::ir

#endif // HAIL_INITALLDIALECTS_H_
