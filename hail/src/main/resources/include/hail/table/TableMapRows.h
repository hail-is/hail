#ifndef HAIL_TABLEMAPROWS_H
#define HAIL_TABLEMAPROWS_H 1

#include "hail/table/PartitionContext.h"

namespace hail {

template<typename Next, typename Mapper>
class TableMapRows {
  private:
    Next next_;
    Mapper mapper_{};

  public:
    using Endpoint = typename Next::Endpoint;
    Endpoint * end() { return next_.end(); }
    PartitionContext * ctx() { return next_.ctx(); }

    void operator()(const char * value) {
      next_(mapper_(ctx()->st_, ctx()->region_.get(), ctx()->globals_, value));
    }

    template<typename ... Args>
    explicit TableMapRows(Args ... args) : next_(args...) { }
};

}

#endif