#include "Dialect/Hail/HailDialect.h"
#include "Dialect/Hail/HailOps.h"

using namespace hail;
using namespace hail::ir;

#include "Dialect/Hail/HailOpsDialect.cpp.inc"

void HailDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/Hail/HailOps.cpp.inc"
      >();
}
