#ifndef HAIL_TABLEEXPLODE_H
#define HAIL_TABLEEXPLODE_H 1

#include "hail/table/PartitionContext.h"
#include "hail/Region.h"

namespace hail {

template<typename Next, typename Exploder>
class TableExplodeRows {
  private:
    Next next_;
    Exploder exploder_{};
    //exploder_.len(st, old_region, it_);
    //exploder_(st, new_region, it_, i);

  public:
    using Endpoint = typename Next::Endpoint;
    Endpoint * end() { return next_.end(); }
    PartitionContext * ctx() { return next_.ctx(); }

    void operator()(const char * value) {
      auto old_region = std::move(ctx()->region_);
      auto len = exploder_.len(ctx()->st_, old_region.get(), value);
      for (int i=0; i<len; ++i) {
        ctx()->new_region();
        ctx()->region_->add_reference_to(old_region);
        next_(exploder_(ctx()->st_, ctx()->region_.get(), value, i));
      }
    }

    template<typename ... Args>
    explicit TableExplodeRows(Args ... args) : next_(args...) { }
};

}

#endif