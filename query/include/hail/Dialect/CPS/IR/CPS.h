#ifndef HAIL_DIALECT_CPS_IR_CPS_H
#define HAIL_DIALECT_CPS_IR_CPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "hail/Dialect/CPS/IR/CPSOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/CPS/IR/CPSOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "hail/Dialect/CPS/IR/CPSOps.h.inc"

#endif // HAIL_DIALECT_CPS_IR_CPS_H
