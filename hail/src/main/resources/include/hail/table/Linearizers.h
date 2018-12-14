#ifndef HAIL_LINEARIZERS_H
#define HAIL_LINEARIZERS_H 1

#include "hail/Region.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace hail {

struct RegionValue {
  RegionPtr region_ = nullptr;
  char const * value_ = nullptr;
  RegionValue(RegionPtr && region, char const * value) : region_(region), value_(value) { }
  RegionValue(nullptr_t) : RegionValue(nullptr, nullptr) { }

  RegionValue() = default;
  RegionValue(RegionValue &rv) : region_(rv.region_), value_(rv.value_) { }
  RegionValue(RegionValue &&rv) : region_(std::move(rv.region_)), value_(rv.value_) { rv = nullptr; }
  RegionValue & operator=(std::nullptr_t) { region_ = nullptr; value_ = nullptr; return *this; }
  RegionValue & operator=(RegionValue &rv) { region_ = rv.region_; value_ = rv.value_; return *this; }
  RegionValue & operator=(RegionValue &&rv) { region_ = rv.region_; value_ = rv.value_; rv = nullptr; return *this; }

  bool operator==(std::nullptr_t) { return value_ == nullptr; }
  bool operator!=(std::nullptr_t) { return value_ != nullptr; }
};

class NestedLinearizerEndpoint {
  private:
    PartitionContext * ctx_;
  public:
    using Endpoint = NestedLinearizerEndpoint;
    Endpoint * end() { return this; }
    PartitionContext * ctx() { return ctx_; }

    std::vector<RegionValue> values_{};
    size_t off_ = 0;

    void operator()(RegionPtr &&region, const char * value) {
      values_.emplace_back(std::move(region), value);
    }

    bool has_value() {
      if (off_ != values_.size()) { return true; }
      values_.clear();
      off_ = 0;
      return false;
    }

    RegionValue && get_next_value() { return std::move(values_[off_++]); }

    NestedLinearizerEndpoint(PartitionContext * ctx) : ctx_(ctx) { }
};

class UnnestedLinearizerEndpoint {
  private:
    PartitionContext * ctx_;
  public:
    using Endpoint = UnnestedLinearizerEndpoint;
    Endpoint * end() { return this; }
    PartitionContext * ctx() { return ctx_; }

    RegionValue value_{};

    void operator()(RegionPtr && region, const char * value) { value_ = { std::move(region), value }; }

    bool has_value() { return value_ != nullptr; }

    RegionValue && get_next_value() { return std::move(value_); }
    UnnestedLinearizerEndpoint(PartitionContext * ctx) : ctx_(ctx) { }
};

template <typename LinearizableConsumer>
class LinearizedPullStream {
  using Linearizer = typename LinearizableConsumer::Endpoint;
  LinearizableConsumer cons_;
  typename LinearizableConsumer::Endpoint * linearizer_ = cons_.end();

  mutable RegionValue value_ {};

  RegionValue & get() const { return value_; }

  bool advance() {
    while (!linearizer_->has_value() && cons_.advance()) {
      cons_.consume();
    }
    value_ = linearizer_->get_next_value();
    return value_ != nullptr;
  }

  public:
    template<typename ... Args>
    explicit LinearizedPullStream(Args&& ... args) :
    cons_(args...) { advance(); }

    class Iterator {
      friend class LinearizedPullStream;
      private:
        LinearizedPullStream * stream_;
        explicit Iterator(LinearizedPullStream * stream) : stream_(stream) { }
      public:
        Iterator() = delete;
        Iterator& operator++() {
          if (!stream_->advance()) {
            stream_ = nullptr;
          }
          return *this;
        }

        friend bool operator==(Iterator const& lhs, Iterator const& rhs) {
          return (lhs.stream_ == rhs.stream_);
        }
        friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
          return !(lhs == rhs);
        }
        RegionValue & operator*() const { return stream_->get(); }
    };

    Iterator begin() { return Iterator(this); }
    Iterator end() { return Iterator(nullptr); }
};

}

#endif