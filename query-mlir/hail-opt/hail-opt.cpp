#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

// dialect includes
#include "Dialect/Hail/HailDialect.h"

int main(int argc, char **argv) {
    mlir::registerAllPasses();
    // FIXME register hail passes

    mlir::DialectRegistry registry;
    registry.insert<hail::ir::HailDialect>();
    registry.insert<mlir::StandardOpsDialect, mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect>();
    // Add the following to include *all* MLIR Core dialects, or selectively
    // include what you need like above. You only need to register dialects that
    // will be *parsed* by the tool, not the one generated
    // registerAllDialects(registry);

    return mlir::asMainReturnCode(
            mlir::MlirOptMain(argc, argv, "Hail optimizer driver\n", registry));
}
