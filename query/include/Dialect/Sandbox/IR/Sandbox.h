#ifndef DIALECT_SANDBOX_SANDBOX_H
#define DIALECT_SANDBOX_SANDBOX_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/Sandbox/IR/SandboxOpsDialect.h.inc"
#include "Dialect/Sandbox/IR/SandboxOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Sandbox/IR/SandboxOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Sandbox/IR/SandboxOps.h.inc"

#endif // DIALECT_SANDBOX_SANDBOX_H
