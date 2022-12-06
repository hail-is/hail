#ifndef HAIL_ANALYSIS_MISSINGNESSANALYSIS_H
#define HAIL_ANALYSIS_MISSINGNESSANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

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
  auto isMissing() const -> bool { return state == Missing; }
  auto isPresent() const -> bool { return state == Present; }
  auto getState() const -> State { return state; }

  void setMissing() const { join(*this, Missing); }
  void setPresent() const { join(*this, Present); }

  auto operator==(MissingnessValue const &rhs) const -> bool { return state == rhs.state; }

  void print(llvm::raw_ostream &os) const;

  static auto getPessimisticValueState(mlir::Value value) -> MissingnessValue { return {Unknown}; }

  static auto join(MissingnessValue const &lhs, MissingnessValue const &rhs) -> MissingnessValue {
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
