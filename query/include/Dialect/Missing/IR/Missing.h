#ifndef DIALECT_MISSING_MISSING_H
#define DIALECT_MISSING_MISSING_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/Missing/IR/MissingOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Missing/IR/MissingOps.h.inc"

#endif // DIALECT_MISSING_MISSING_H
