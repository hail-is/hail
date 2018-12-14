#ifndef HAIL_TABLEEMIT_H
#define HAIL_TABLEEMIT_H 1
#include "hail/RegionPool.h"

namespace hail {

struct PartitionContext {
  NativeStatus * st_;
  const char * globals_;
  RegionPool pool_{};

  PartitionContext(NativeStatus * st, const char * globals) : st_(st), globals_(globals) { }
  PartitionContext(NativeStatus * st) : PartitionContext(st, nullptr) { }
  PartitionContext() : PartitionContext(nullptr) { }
};

}

#endif