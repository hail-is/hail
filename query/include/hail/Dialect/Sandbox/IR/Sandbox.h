#ifndef HAIL_DIALECT_SANDBOX_IR_SANDBOX_H
#define HAIL_DIALECT_SANDBOX_IR_SANDBOX_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "hail/Dialect/Sandbox/IR/SandboxOpsDialect.h.inc"
#include "hail/Dialect/Sandbox/IR/SandboxOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hail/Dialect/Sandbox/IR/SandboxOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "hail/Dialect/Sandbox/IR/SandboxOps.h.inc"

#endif // HAIL_DIALECT_SANDBOX_IR_SANDBOX_H
