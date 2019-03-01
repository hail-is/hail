#ifndef HAIL_ARRAYSORTER_H
#define HAIL_ARRAYSORTER_H 1

#include <vector>
#include <algorithm>

namespace hail {

template <typename ArrayBuilder, typename LessThan>
class ArraySorter {
  public:
    using ArrayImpl = typename ArrayBuilder::ArrayImpl;
    using T = typename ArrayImpl::T;

    struct SortableElem {
      T elem_;
      LessThan lt_;

      SortableElem(T elem) : elem_(elem) { }
      friend bool operator<(const SortableElem& lhs, const SortableElem& rhs) { return lhs.lt_(lhs.elem_, rhs.elem_); }
    };

    std::vector<SortableElem> non_missing_;
    int n_missing_;

    ArraySorter(const char * array) : non_missing_() {
      int len = ArrayImpl::load_length(array);
      non_missing_.reserve(len);
      for (int i=0; i<len; ++i) {
        if (!ArrayImpl::is_element_missing(array, i)) {
          non_missing_.emplace_back(ArrayImpl::load_element(array, i));
        }
      }
      n_missing_ = len - non_missing_.size();
    }

    void sort() { std::stable_sort(non_missing_.begin(), non_missing_.end()); }

    template<typename IsEqual>
    void distinct() {
      IsEqual eq_;
      for (auto it = ++non_missing_.begin(); it != non_missing_.end(); ) {
        if (eq_((*it).elem_, (*(it - 1)).elem_)) {
            it = non_missing_.erase(it);
        } else { ++it; }
      }
      if (n_missing_ != 0) { n_missing_ = 1; }
    }

    char * to_region(Region * region) {
      int len = (int) non_missing_.size() + n_missing_;
      ArrayBuilder ab { len, region };
      ab.clear_missing_bits();
      for (int i=0; i<non_missing_.size(); ++i) {
        ab.set_element(i, non_missing_[i].elem_);
      }
      for (int i=non_missing_.size(); i<len; ++i) { ab.set_missing(i); }
      return ab.offset();
    }

    char * to_region(RegionPtr region) { return to_region(region.get()); }
};

}

#endif