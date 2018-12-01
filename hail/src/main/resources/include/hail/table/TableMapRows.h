#ifndef HAIL_TABLEMAPROWS_H
#define HAIL_TABLEMAPROWS_H 1

#include "hail/table/TableEmit.h"
#include "hail/NativeStatus.h"
#include "hail/Utils.h"

namespace hail {

template<typename Prev, typename Mapper>
class TableMapRows {
  friend class TablePartitionRange<TableMapRows>;
  private:
    typename Prev::Iterator it_;
    typename Prev::Iterator end_;
    PartitionContext * ctx_;
    Mapper mapper_{};
    char const * value_ = nullptr;
    bool advance() {
      ++it_;
      if (LIKELY(it_ != end_)) {
        value_ = mapper_(ctx_->st_, ctx_->region_.get(), ctx_->globals_, *it_);
      } else {
        value_ = nullptr;
      }
      return (value_ != nullptr);
    }

  public:
    TableMapRows(PartitionContext * ctx, Prev & prev) :
    it_(prev.begin()),
    end_(prev.end()),
    ctx_(ctx) {
      if (LIKELY(it_ != end_)) {
        value_ = mapper_(ctx_->st_, ctx_->region_.get(), ctx_->globals_, *it_);
      }
    }
};

}

#endif