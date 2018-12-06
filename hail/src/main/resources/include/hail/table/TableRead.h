#ifndef HAIL_TABLEREAD_H
#define HAIL_TABLEREAD_H 1

#include "hail/table/PartitionContext.h"

namespace hail {

template<typename Next, typename Decoder>
class TableNativeRead {
  private:
    Next next_;
    Decoder dec_;
  public:
    using Endpoint = typename Next::Endpoint;
    Endpoint * end() { return next_.end(); }
    PartitionContext * ctx() { return next_.ctx(); }

    void consume() {
      next_(dec_.decode_row(ctx()->region_.get()));
    }

    bool advance() {
      ctx()->new_region();
      return dec_.decode_byte();
    }

    template<typename ... Args>
    explicit TableNativeRead(Decoder &&dec, Args ... args) :
    next_(args...),
    dec_(std::move(dec)) { }
};

}

#endif
