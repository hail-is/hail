#ifndef HAIL_LINEARIZERS_H
#define HAIL_LINEARIZERS_H 1

#include "hail/Region.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace hail {

class NestedLinearizerEndpoint {
  private:
    PartitionContext * ctx_;
  public:
    using Endpoint = NestedLinearizerEndpoint;
    Endpoint * end() { return this; }
    PartitionContext * ctx() { return ctx_; }

    std::vector<RegionPtr> regions_{};
    std::vector<char const *> values_{};
    size_t off_ = 0;

    void operator()(const char * value) {
      regions_.push_back(ctx_->region_);
      values_.push_back(value);
    }

    bool has_value() {
      if (off_ == regions_.size()) {
        regions_.clear();
        values_.clear();
        off_ = 0;
        return false;
      }
      return true;
    }

    char const * get_next_value() {
      if (off_ > 0) { regions_[off_ - 1] = nullptr; }
      ctx_->region_ = regions_[off_];
      return values_[off_++];
    }

    NestedLinearizerEndpoint(PartitionContext * ctx) : ctx_(ctx) { }
};

class UnnestedLinearizerEndpoint {
  private:
    PartitionContext * ctx_;
  public:
    using Endpoint = UnnestedLinearizerEndpoint;
    Endpoint * end() { return this; }
    PartitionContext * ctx() { return ctx_; }

    char const * value_ = nullptr;

    void operator()(const char * value) {
      value_ = value;
    }

    bool has_value() { return value_ != nullptr; }
    char const * get_next_value() {
      auto v = value_;
      value_ = nullptr;
      return v;
    }

    UnnestedLinearizerEndpoint(PartitionContext * ctx) : ctx_(ctx) { }
};

template <typename LinearizableConsumer>
class LinearizedPullStream {
  using Linearizer = typename LinearizableConsumer::Endpoint;
  LinearizableConsumer cons_;
  typename LinearizableConsumer::Endpoint * linearizer_ = cons_.end();

  char const * get() { return linearizer_->get_next_value(); }

  void advance() {
    while (!linearizer_->has_value() && cons_.advance()) {
      cons_.consume();
    }
  }

  public:
    explicit LinearizedPullStream(LinearizableConsumer &&cons) :
    cons_(cons) { advance(); }

    class Iterator {
      friend class LinearizedPullStream;
      private:
        LinearizedPullStream * stream_;
        explicit Iterator(LinearizedPullStream * stream) : stream_(stream) { }
      public:
        Iterator() = delete;
        Iterator& operator++() {
          stream_->advance();
          if (!stream_->linearizer_->has_value()) { stream_ = nullptr; }
          return *this;
        }

        friend bool operator==(Iterator const& lhs, Iterator const& rhs) {
          return (lhs.stream_ == rhs.stream_);
        }
        friend bool operator!=(Iterator const& lhs, Iterator const& rhs) {
          return !(lhs == rhs);
        }
        char const* operator*() const { return stream_->get(); }
    };

    Iterator begin() { return Iterator(linearizer_->has_value() ? this : nullptr); }
    Iterator end() { return Iterator(nullptr); }
};

}

#endif