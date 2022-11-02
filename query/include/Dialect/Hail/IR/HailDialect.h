#ifndef DIALECT_HAIL_HAILDIALECT_H
#define DIALECT_HAIL_HAILDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Hail/IR/HailOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Hail/IR/HailOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Hail/IR/HailOps.h.inc"

#endif // DIALECT_HAIL_HAILDIALECT_H
