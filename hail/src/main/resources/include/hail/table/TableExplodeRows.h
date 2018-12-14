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

    void operator()(RegionPtr &&region, const char * value) {
      auto len = exploder_.len(ctx()->st_, region.get(), value);
      for (int i=0; i<len; ++i) {
        auto new_region = ctx()->pool_.get_region();
        new_region->add_reference_to(region);
        next_(std::move(new_region), exploder_(ctx()->st_, new_region.get(), value, i));
      }
    }

    template<typename ... Args>
    explicit TableExplodeRows(Args ... args) : next_(args...) { }
};

}

#endif