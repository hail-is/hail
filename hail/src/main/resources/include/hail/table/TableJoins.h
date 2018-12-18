#ifndef HAIL_TABLEJOINS_H
#define HAIL_TABLEJOINS_H 1

#include "hail/table/PartitionContext.h"
#include "hail/Region.h"
#include "hail/table/Linearizers.h"
#include <iostream>
#include <stdexcept>

namespace hail {

template<typename Consumer, typename RightPartitionPullStream, typename JoinF>
class TableLeftJoinRightDistinct {
  private:
    Consumer next_;
    JoinF joinf_{};
    // int joinf_.compare(rowL, rowR);
    // const char * joinf_(new_region, rowL, rowR);
    typename RightPartitionPullStream::Iterator it_;
    typename RightPartitionPullStream::Iterator end_;
    RegionValue row_ {std::move(*it_)};
  public:
    using Endpoint = typename Consumer::Endpoint;
    Endpoint * end() { return next_.end(); }
    PartitionContext * ctx() { return next_.ctx(); }

    void operator()(RegionPtr &&region, const char * value) {
      if (it_ != end_ && (row_ == nullptr || joinf_.compare(value, row_.value_) > 0)) {
        while (++it_ != end_ && joinf_.compare(value, (*it_).value_) > 0) { }
        row_ = (it_ == end_) ? nullptr : std::move(*it_);
      }
      if (row_.value_ != nullptr && joinf_.compare(value, row_.value_) == 0) {
        auto new_region = ctx()->pool_.get_region();
        new_region->add_reference_to(std::move(region));
        new_region->add_reference_to(row_.region_);
        next_(std::move(new_region), joinf_(new_region.get(), value, row_.value_));
      } else {
        next_(std::move(region), joinf_(region.get(), value, nullptr));
      }
    }

    template<typename ... Args>
    explicit TableLeftJoinRightDistinct(RightPartitionPullStream * stream, Args ... args) :
    next_(args...),
    it_(stream->begin()),
    end_(stream->end()) { }
};

}

#endif