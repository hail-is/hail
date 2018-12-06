#ifndef HAIL_TABLEWRITE_H
#define HAIL_TABLEWRITE_H 1

#include "hail/table/PartitionContext.h"
#include "hail/Encoder.h"

namespace hail {

template<typename Encoder>
class TableNativeWrite : public Encoder {
    PartitionContext * ctx_;
  public:
    using Encoder::encode_byte;
    using Encoder::encode_row;
    using Encoder::flush;

    using Endpoint = TableNativeWrite;
    Endpoint * end() { return this; }
    PartitionContext * ctx() { return ctx_; }

    void operator()(const char * value) {
      encode_byte(1);
      encode_row(value);
    }

    using Encoder::Encoder;
    template<typename ... Args>
    explicit TableNativeWrite(PartitionContext * ctx, Args ... args) :
    Encoder(args...), ctx_(ctx) { }

};

}

#endif