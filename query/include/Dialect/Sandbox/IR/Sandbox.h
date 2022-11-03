#ifndef DIALECT_SANDBOX_SANDBOX_H
#define DIALECT_SANDBOX_SANDBOX_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Sandbox/IR/SandboxOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Sandbox/IR/SandboxOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/Sandbox/IR/SandboxOps.h.inc"

#endif // DIALECT_SANDBOX_SANDBOX_H
