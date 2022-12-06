#include "./PassDetail.h"

#include "hail/Analysis/MissingnessAnalysis.h"
#include "hail/Analysis/MissingnessAwareConstantPropagationAnalysis.h"
#include "hail/Support/MLIR.h"
#include "hail/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

using namespace hail::ir;

static void rewrite(mlir::DataFlowSolver &solver, Operation *root) {
  root->walk([&solver, root](Operation *op) {
    auto *context = root->getContext();
    auto builder = Builder(root->getContext());
    SmallVector<Attribute> annotations;

    annotations.reserve(op->getNumOperands());
    for (Value const result : op->getResults()) {
      auto *missingnessState =
          solver.getOrCreateState<mlir::dataflow::Lattice<hail::ir::MissingnessValue>>(result);
      auto *constState =
          solver.getOrCreateState<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(result);

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
      op->setAttr("missing.result_states", ArrayAttr::get(context, annotations));

    annotations.clear();
    if (op->getNumRegions() > 0 && op->getRegion(0).getNumArguments() > 0) {
      auto &region = op->getRegion(0);
      annotations.reserve(region.getNumArguments());
      for (Value const arg : region.getArguments()) {
        auto *missingnessState =
            solver.getOrCreateState<mlir::dataflow::Lattice<hail::ir::MissingnessValue>>(arg);
        auto *constState =
            solver.getOrCreateState<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>>(arg);

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

      op->setAttr("missing.region_arg_states", ArrayAttr::get(context, annotations));
    }
  });
}

namespace {
struct TestMissingnessAnalysisPass
    : public TestMissingnessAnalysisBase<TestMissingnessAnalysisPass> {
  void runOnOperation() override;
};
} // namespace

void TestMissingnessAnalysisPass::runOnOperation() {
  Operation *op = getOperation();

  mlir::DataFlowSolver solver;
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<MissingnessAwareConstantPropagation>();
  solver.load<MissingnessAnalysis>();
  if (failed(solver.initializeAndRun(op)))
    return signalPassFailure();
  rewrite(solver, op);
}

auto hail::ir::createTestMissingnessAnalysisPass() -> std::unique_ptr<Pass> {
  return std::make_unique<TestMissingnessAnalysisPass>();
}
