#ifndef HAIL_TABLEFILTERROWS_H
#define HAIL_TABLEFILTERROWS_H 1

#include "hail/table/PartitionContext.h"

namespace hail {

template<typename Next, typename Filter>
class TableFilterRows {
  private:
    Next next_;
    Filter filter_{};

  public:
    using Endpoint = typename Next::Endpoint;
    Endpoint * end() { return next_.end(); }
    PartitionContext * ctx() { return next_.ctx(); }

    void operator()(const char * value) {
      if (filter_(ctx()->st_, ctx()->region_.get(), ctx()->globals_, value)) {
        next_(value);
      }
    }

    template<typename ... Args>
    explicit TableFilterRows(Args ... args) : next_(args...) { }
};

}

#endif