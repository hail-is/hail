#ifndef CONVERSION_PASSES_H
#define CONVERSION_PASSES_H

#include "Conversion/LowerSandbox/LowerSandbox.h"

namespace hail {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace hail

#endif // CONVERSION_PASSES_H