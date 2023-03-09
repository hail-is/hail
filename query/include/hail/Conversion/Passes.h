#ifndef HAIL_CONVERSION_PASSES_H
#define HAIL_CONVERSION_PASSES_H

#include "hail/Conversion/CPSToCF/CPSToCF.h"
#include "hail/Conversion/LowerOption/LowerOption.h"
#include "hail/Conversion/LowerSandbox/LowerSandbox.h"
#include "hail/Conversion/LowerToLLVM/LowerToLLVM.h"
#include "hail/Conversion/OptionToGenericOption/OptionToGenericOption.h"
#include "mlir/Pass/PassRegistry.h"

namespace hail {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "hail/Conversion/Passes.h.inc"

} // namespace hail

#endif // HAIL_CONVERSION_PASSES_H
