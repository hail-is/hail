#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "hail/InitAllDialects.h"
#include "hail/InitAllPasses.h"

auto main(int argc, char **argv) -> int {
  // mlir::registerAllPasses();

  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertLinalgToStandardPass();
  mlir::registerConvertTensorToLinalgPass();
  mlir::registerConvertVectorToSCFPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();

  // Dialect passes
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerLinalgLowerToAffineLoopsPass();
  mlir::registerLinalgLowerToLoopsPass();

  // Hail passes
  hail::ir::registerAllPasses();

  // Dialects
  mlir::DialectRegistry registry;
  hail::ir::registerAllDialects(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Hail optimizer driver\n", registry));
}
