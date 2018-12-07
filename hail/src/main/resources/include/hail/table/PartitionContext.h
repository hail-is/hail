#ifndef HAIL_TABLEEMIT_H
#define HAIL_TABLEEMIT_H 1
#include "hail/RegionPool.h"

//class PushConsumer {
//  static constexpr bool is_linear_;
//  static constexpr bool is_nested_;
//  void operator()(PartitionContext * ctx, const char * value);
//  void end();
//}

//class PushProducer {
//    (PartitionContext * ctx_;)
//    void consume();
//    bool advance();
//    void end();
//}

namespace hail {

struct PartitionContext {
  const char * globals_;
  RegionPool pool_{};

  PartitionContext(const char * globals) : globals_(globals) { }
  PartitionContext() : PartitionContext(nullptr) { }
};

}

#endif