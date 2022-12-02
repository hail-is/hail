#ifndef DIALECT_OPTION_OPTION_H
#define DIALECT_OPTION_OPTION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/CPS/IR/CPS.h"

#include "Dialect/Option/IR/OptionOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Option/IR/OptionOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Option/IR/OptionOps.h.inc"

#endif // DIALECT_OPTION_OPTION_H
