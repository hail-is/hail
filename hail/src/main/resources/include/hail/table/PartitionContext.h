#ifndef HAIL_TABLEEMIT_H
#define HAIL_TABLEEMIT_H 1
#include "hail/RegionPool.h"

namespace hail {

struct PartitionContext {
  const char * globals_;
  RegionPool pool_{};

  PartitionContext(const char * globals) : globals_(globals) { }
  PartitionContext() : PartitionContext(nullptr) { }
};

}

#endif