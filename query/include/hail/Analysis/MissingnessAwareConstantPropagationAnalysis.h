//===- ConstantPropagationAnalysis.h - Constant propagation analysis ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant propagation analysis. In this file are defined
// the lattice value class that represents constant values in the program and
// a sparse constant propagation analysis that uses operation folders to
// speculate about constant values in the program.
//
//===----------------------------------------------------------------------===//

#ifndef HAIL_ANALYSIS_MISSINGNESSAWARECONSTANTPROPAGATIONANALYSIS_H
#define HAIL_ANALYSIS_MISSINGNESSAWARECONSTANTPROPAGATIONANALYSIS_H

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace hail::ir {

//===----------------------------------------------------------------------===//
// SparseConstantPropagation
//===----------------------------------------------------------------------===//

/// This analysis implements sparse constant propagation, which attempts to
/// determine constant-valued results for operations using constant-valued
/// operands, by speculatively folding operations. When combined with dead-code
/// analysis, this becomes sparse conditional constant propagation (SCCP).
///
/// It is intended to also be combined with requiredness analysis. For that to
/// be sound, it must satisfy the invariant that whenever a value is inferred to
/// me missing, its inferred constant value is uninitialized.
class MissingnessAwareConstantPropagation
    : public mlir::dataflow::SparseDataFlowAnalysis<
          mlir::dataflow::Lattice<mlir::dataflow::ConstantValue>> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(
      mlir::Operation *op,
      llvm::ArrayRef<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue> const *> operands,
      llvm::ArrayRef<mlir::dataflow::Lattice<mlir::dataflow::ConstantValue> *> results) override;
};

} // namespace hail::ir

#endif // HAIL_ANALYSIS_MISSINGNESSAWARECONSTANTPROPAGATIONANALYSIS_H
