#ifndef HAIL_TRANSFORMS_PASSES
#define HAIL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def TestMissingnessAnalysis : Pass<"test-missingness-analysis"> {
  let summary = "annotate with results of missingness analysis";
  let description = [{
    This pass adds anotations recording the inferences of the missingness aware
    Sparse Conditional Constant Propagation.
  }];
  let constructor = "hail::ir::createTestMissingnessAnalysisPass()";
}

#endif // HAIL_TRANSFORMS_PASSES
