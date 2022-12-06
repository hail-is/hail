#ifndef HAIL_DIALECT_OPTION_IR_OPTION_H
#define HAIL_DIALECT_OPTION_IR_OPTION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hail/Dialect/CPS/IR/CPS.h"

#include "hail/Dialect/Option/IR/OptionOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/Option/IR/OptionOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "hail/Dialect/Option/IR/OptionOps.h.inc"

#endif // HAIL_DIALECT_OPTION_IR_OPTION_H
