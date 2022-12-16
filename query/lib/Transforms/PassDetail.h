// NOLINTNEXTLINE(llvm-header-guard)
#ifndef HAIL_TRANSFORMS_PASSDETAIL_H_
#define HAIL_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace hail::ir {
#define GEN_PASS_CLASSES
#include "hail/Transforms/Passes.h.inc"
} // namespace hail::ir

#endif // HAIL_TRANSFORMS_PASSDETAIL_H_
