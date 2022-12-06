#ifndef HAIL_ANALYSIS_MISSINGNESSANALYSIS_H
#define HAIL_ANALYSIS_MISSINGNESSANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace hail::ir {

//===----------------------------------------------------------------------===//
// MissingnessValue
//===----------------------------------------------------------------------===//

class MissingnessValue {
public:
  enum State {
    Present, // value is always present
    Unknown, // value might be present or missing
    Missing  // value is always missing
  };

  MissingnessValue(State state) : state(state) {}
  bool isMissing() const { return state == Missing; }
  bool isPresent() const { return state == Present; }
  State getState() const { return state; }

  void setMissing() { join(*this, Missing); }
  void setPresent() { join(*this, Present); }

  bool operator==(MissingnessValue const &rhs) const { return state == rhs.state; }

  void print(llvm::raw_ostream &os) const;

  static MissingnessValue getPessimisticValueState(mlir::Value value) { return {Unknown}; }

  static MissingnessValue join(MissingnessValue const &lhs, MissingnessValue const &rhs) {
    return lhs == rhs ? lhs : MissingnessValue(Unknown);
  }

private:
  State state;
};

//===----------------------------------------------------------------------===//
// SparseConstantPropagation
//===----------------------------------------------------------------------===//

/// This analysis implements sparse constant propagation, which attempts to
/// determine constant-valued results for operations using constant-valued
/// operands, by speculatively folding operations. When combined with dead-code
/// analysis, this becomes sparse conditional constant propagation (SCCP).
class MissingnessAnalysis
    : public mlir::dataflow::SparseDataFlowAnalysis<mlir::dataflow::Lattice<MissingnessValue>> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(mlir::Operation *op,
                      llvm::ArrayRef<mlir::dataflow::Lattice<MissingnessValue> const *> operands,
                      llvm::ArrayRef<mlir::dataflow::Lattice<MissingnessValue> *> results) override;
};

} // namespace hail::ir

#endif // HAIL_ANALYSIS_MISSINGNESSANALYSIS_H
