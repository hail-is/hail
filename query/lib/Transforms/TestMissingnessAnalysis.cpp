#include "PassDetail.h"
#include "Transforms/Passes.h"
#include "Analysis/MissingnessAnalysis.h"
#include "Analysis/MissingnessAwareConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace hail::ir;

static void rewrite(mlir::DataFlowSolver &solver, mlir::Operation *root) {
  root->walk([&solver, root](mlir::Operation *op) {
    auto context = root->getContext();
    auto builder = mlir::Builder(root->getContext());
    llvm::SmallVector<mlir::Attribute> annotations;

    annotations.reserve(op->getNumOperands());
    for (mlir::Value result : op->getResults()) {
      auto *missingnessState = solver.getOrCreateState<mlir::dataflow::Lattice<hail::ir::MissingnessValue>>(result);
      auto *constState = solver.getOrCreateState<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(result);

      if (missingnessState->isUninitialized())
        continue;

      if (missingnessState->getValue().isMissing()) {
        annotations.push_back(builder.getStringAttr("Missing"));
      } else if (auto constVal = constState->getValue().getConstantValue()) {
        annotations.push_back(constVal);
      } else if (missingnessState->getValue().isPresent()) {
        annotations.push_back(builder.getStringAttr("Present"));
      } else {
        annotations.push_back(builder.getStringAttr("?"));
      }
    }

    if (op->getNumResults() > 0)
      op->setAttr("missing.result_states", mlir::ArrayAttr::get(context, annotations));

    annotations.clear();
    if (op->getNumRegions() > 0 && op->getRegion(0).getNumArguments() > 0) {
      auto &region = op->getRegion(0);
      annotations.reserve(region.getNumArguments());
      for (mlir::Value arg : region.getArguments()) {
        auto *missingnessState = solver.getOrCreateState<mlir::dataflow::Lattice<hail::ir::MissingnessValue>>(arg);
        auto *constState = solver.getOrCreateState<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(arg);

        if (missingnessState->isUninitialized())
          continue;

        if (missingnessState->getValue().isMissing()) {
          annotations.push_back(builder.getStringAttr("Missing"));
        } else if (auto constVal = constState->getValue().getConstantValue()) {
          annotations.push_back(constVal);
        } else if (missingnessState->getValue().isPresent()) {
          annotations.push_back(builder.getStringAttr("Present"));
        } else {
          annotations.push_back(builder.getStringAttr("?"));
        }
      }

      op->setAttr("missing.region_arg_states", mlir::ArrayAttr::get(context, annotations));
    }
  });
}

namespace {
struct TestMissingnessAnalysisPass : public TestMissingnessAnalysisBase<TestMissingnessAnalysisPass> {
  void runOnOperation() override;
};
} // namespace

void TestMissingnessAnalysisPass::runOnOperation() {
  mlir::Operation *op = getOperation();

  mlir::DataFlowSolver solver;
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<MissingnessAwareConstantPropagation>();
  solver.load<MissingnessAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  rewrite(solver, op);
}

std::unique_ptr<mlir::Pass> hail::ir::createTestMissingnessAnalysisPass() {
  return std::make_unique<TestMissingnessAnalysisPass>();
}
