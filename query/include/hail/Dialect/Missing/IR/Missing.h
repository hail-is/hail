#ifndef HAIL_DIALECT_MISSING_IR_MISSING_H
#define HAIL_DIALECT_MISSING_IR_MISSING_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "hail/Dialect/Missing/IR/MissingOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "hail/Dialect/Missing/IR/MissingOps.h.inc"

#endif // HAIL_DIALECT_MISSING_IR_MISSING_H
