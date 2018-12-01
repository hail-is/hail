#ifndef HAIL_TABLEEMIT_H
#define HAIL_TABLEEMIT_H 1
#include "hail/RegionPool.h"
#include "hail/Utils.h"
#include <string>

namespace hail {

struct PartitionContext {
  NativeStatus * st_;
  const char * globals_;
  RegionPool pool_{};
  RegionPtr region_;
  void new_region() { region_ = pool_.get_region(); }

  PartitionContext(NativeStatus * st, const char * globals) : st_(st), globals_(globals), region_(pool_.get_region()) { }
  PartitionContext(NativeStatus * st) : PartitionContext(st, nullptr) { }
};

template <typename TableIRPartition>
class TablePartitionRange : public TableIRPartition {
  private:
    using TableIRPartition::value_;
//    bool advance();
    char const* get() const { return value_; }
  public:
    class Iterator {
      friend class TablePartitionRange;
      private:
        TablePartitionRange * range_;
        explicit Iterator(TablePartitionRange * range) : range_(range) { }
      public:
        Iterator& operator++() {
          if (range_ != nullptr && !(range_->advance())) {
            range_ = nullptr;
          }
          return *this;
        }
        char const* operator*() const { return range_->get(); }
        friend bool operator==(Iterator const& lhs, Iterator const& rhs) {
          return (lhs.range_ == rhs.range_);
        }
        friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
          return !(lhs == rhs);
        }
    };
    using TableIRPartition::TableIRPartition;
    Iterator begin() { return Iterator(this); }
    Iterator end() { return Iterator(nullptr); }
};

}

#endif