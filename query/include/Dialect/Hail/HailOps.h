#ifndef DIALECT_HAIL_HAILOPS_H
#define DIALECT_HAIL_HAILOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Dialect/Hail/HailOps.h.inc"

#endif // DIALECT_HAIL_HAILOPS_H
