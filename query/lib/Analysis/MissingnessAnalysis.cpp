#include "hail/Analysis/MissingnessAnalysis.h"
#include "hail/Dialect/Missing/IR/Missing.h"
#include "hail/Support/MLIR.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "missingness-analysis"

using namespace hail::ir;

//===----------------------------------------------------------------------===//
// MissingnessValue
//===----------------------------------------------------------------------===//

void MissingnessValue::print(llvm::raw_ostream &os) const {
  if (state == State::Missing)
    os << "<Missing>";
  else if (state == State::Present)
    os << "<Present>";
  else
    os << "<Unknown>";
}

//===----------------------------------------------------------------------===//
// MissingnessAnalysis
//===----------------------------------------------------------------------===//

void MissingnessAnalysis::visitOperation(
    Operation *op, ArrayRef<mlir::dataflow::Lattice<MissingnessValue> const *> operands,
    ArrayRef<mlir::dataflow::Lattice<MissingnessValue> *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Missingness: Visiting operation: " << *op << "\n");

  // FIXME: move missingness op semantics to an interface
  if (auto missingOp = dyn_cast<MissingOp>(op)) {
    propagateIfChanged(results.front(), results.front()->join({MissingnessValue::Missing}));
    return;
  };

  // By default, operations are strict: if any operand is missing, all results are missing
  MissingnessValue::State operandsState{};
  for (auto const *lattice : operands) {
    operandsState = std::max(operandsState, lattice->getValue().getState());
  }
  for (auto *result : results) {
    auto changed = result->join({operandsState});
    LLVM_DEBUG(llvm::dbgs() << " result: "; result->print(llvm::dbgs()); llvm::dbgs() << "\n");
    propagateIfChanged(result, changed);
  };
}
