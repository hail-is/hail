#ifndef HAIL_TABLEEXPLODE_H
#define HAIL_TABLEEXPLODE_H 1

#include "hail/table/TableEmit.h"
#include "hail/NativeStatus.h"
#include "hail/Region.h"
#include "hail/Upcalls.h"
#include <string>

namespace hail {

template<typename Prev, typename Exploder>
class TableExplodeRows {
  friend class TablePartitionRange<TableExplodeRows>;
  private:
    UpcallEnv up_{};
    typename Prev::Iterator it_;
    typename Prev::Iterator end_;
    PartitionContext * ctx_;
    char const * value_ = nullptr;
    size_t len_ = 0;
    size_t i_ = 0;
    char const * exploded_ = nullptr;
    Exploder exploder_{};
    RegionPtr old_region_;
    //exploder_.len(st, old_region, it_);
    //exploder_(st, new_region, it_, i);

    void calc_length() {
      len_ = exploder_.len(ctx_->st_, ctx_->region_.get(), *it_);
      i_ = 0;
      old_region_ = ctx_->region_;
    }

    bool advance() {
      while (i_ == len_) {
        ++it_;
        if (it_ == end_) {
          value_ = nullptr;
          return false;
        }
        calc_length();
      }
      ctx_->new_region();
      ctx_->region_->add_reference_to(old_region_);
      value_ = exploder_(ctx_->st_, ctx_->region_.get(), *it_, i_);
      ++i_;
      return true;
    }

  public:
    TableExplodeRows(PartitionContext * ctx, Prev & prev) :
    it_(prev.begin()),
    end_(prev.end()),
    ctx_(ctx),
    old_region_(ctx->region_) {
      if (it_ != end_) {
        calc_length();
        advance();
      }
    }
};

}

#endif