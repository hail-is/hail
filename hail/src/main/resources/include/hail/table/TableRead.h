#ifndef HAIL_TABLEREAD_H
#define HAIL_TABLEREAD_H 1

#include "hail/table/TableEmit.h"
#include "hail/NativeStatus.h"

namespace hail {

template<typename Decoder>
class TableNativeRead {
  friend class TablePartitionRange<TableNativeRead>;
  private:
    Decoder dec_;
    PartitionContext * ctx_;
    char const * value_ = nullptr;
    bool advance() {
      if (dec_.decode_byte()) {
        ctx_->new_region();
        value_ = dec_.decode_row(ctx_->region_.get());
      } else {
        value_ = nullptr;
      }
      return (value_ != nullptr);
    }

  public:
    TableNativeRead(Decoder dec, PartitionContext * ctx) : dec_(dec), ctx_(ctx) {
      if (dec_.decode_byte()) {
        value_ = dec_.decode_row(ctx_->region_.get());
      }
    }
};

}

#endif
