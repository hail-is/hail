#ifndef HAIL_ARRAYBUILDER_H
#define HAIL_ARRAYBUILDER_H 1

#include "hail/Utils.h"
#include "hail/Region.h"

namespace hail {

template<bool elem_required, size_t elem_size, size_t elem_align, size_t array_align>
class BaseArrayBuilder {
  private:
    int len_;
    char * off_;
    char * elem_addr_;
    
  public:
    using ArrayImpl = BaseArrayImpl<elem_required, elem_size, elem_align>;

    BaseArrayBuilder(int len, Region * region) :
    len_(len), off_(nullptr), elem_addr_(nullptr) {
      int elem_off = ArrayImpl::elements_offset(len_);
      off_ = region->allocate(array_align, elem_off + len_ * ArrayImpl::array_elem_size);
      elem_addr_ = off_ + elem_off;
      store_int(off_, len_);
    }

    BaseArrayBuilder(int len, RegionPtr region) : BaseArrayBuilder(len, region.get()) { }

    void clear_missing_bits() {
      if (!elem_required) { memset(off_ + 4, 0, n_missing_bytes(len_)); }
    }

    void set_missing(int idx) {
      if (elem_required) { throw FatalError("Required array element cannot be missing."); }
      set_bit(off_ + 4, idx);
    }

    char * element_address(int idx) const { return elem_addr_ + idx * ArrayImpl::array_elem_size; }

    char * offset() const { return off_; }
};

template<typename ElemT, bool elem_required, size_t elem_size, size_t elem_align, size_t array_align>
class ArrayLoadBuilder : public BaseArrayBuilder<elem_required, elem_size, elem_align, array_align> {
  public:
    using ArrayImpl = ArrayLoadImpl<ElemT, elem_required, elem_size, elem_align>;
    using T = ElemT;
    using Base = BaseArrayBuilder<elem_required, elem_size, elem_align, array_align>;
    void set_element(int idx, ElemT elem) {
      *reinterpret_cast<ElemT *>(Base::element_address(idx)) = elem;
    }
    using Base::Base;
};

template<bool elem_required, size_t elem_size, size_t elem_align, size_t array_align>
class ArrayAddrBuilder : public BaseArrayBuilder<elem_required, elem_size, elem_align, array_align> {
  public:
    using ArrayImpl = ArrayAddrImpl<elem_required, elem_size, elem_align>;
    using T = char *;
    using Base = BaseArrayBuilder<elem_required, elem_size, elem_align, array_align>;
    void set_element(int idx, const char * elem) {
      memcpy(Base::element_address(idx), elem, elem_size);
    }
    using Base::Base;
};

}

#endif