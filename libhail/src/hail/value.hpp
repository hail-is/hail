#ifndef HAIL_VALUE_HPP_INCLUDED
#define HAIL_VALUE_HPP_INCLUDED 1

#include <memory>

#include <hail/allocators.hpp>
#include <hail/tunion.hpp>
#include <hail/ptype.hpp>

namespace hail {

class FormatStream;
class Value;

struct StrData {
  uint32_t size;
  char data[1];
};

class StrValue {
  friend class Value;

  const PStr *ptype;
  std::shared_ptr<ArenaAllocator> region;
  StrData *p;

public:
  StrValue(const PStr *ptype, std::shared_ptr<ArenaAllocator> region, StrData *p)
    : ptype(ptype),
      region(std::move(region)),
      p(p) {}

  const PStr *get_ptype() const { return ptype; }
  size_t get_size() const { return p->size; }
  char *get_data() { return p->data; }

  inline operator Value() const;
};

inline bool
get_bit(char *p, size_t i) {
  return *(p + (i >> 8)) & (1 << (i & 7));
}

inline void
set_bit(char *p, size_t i, bool b) {
  char *bp = p + (i >> 8);
  int s = (i & 7);
  *bp = (*bp & ~(1 << s)) | (static_cast<int>(b) << s);
}

struct ArrayData {
  uint32_t size;
  char *missing_bits;
  char *elements;
};

class ArrayValue {
  friend class Value;

  const PArray *ptype;
  std::shared_ptr<ArenaAllocator> region;
  ArrayData *p;
  size_t size;
  char *missing_bits;
  char *elements;

public:
  ArrayValue(const PArray *ptype, std::shared_ptr<ArenaAllocator> region, ArrayData *p)
    : ptype(ptype),
      region(std::move(region)),
      p(p),
      size(p->size),
      missing_bits(p->missing_bits),
      elements(p->elements) {}

  const PArray *get_ptype() const { return ptype; }
  size_t get_size() const { return size; }
  bool get_element_present(size_t i) const {
    return get_bit(missing_bits, i);
  }
  void set_element_present(size_t i, bool present) {
    set_bit(missing_bits, i, present);
  }
  inline Value get_element(size_t i) const;
  void set_element(size_t i, const Value &new_element);

  inline operator Value() const;
};

class TupleValue {
  friend class Value;
  
  const PTuple *ptype;
  std::shared_ptr<ArenaAllocator> region;
  char *p;

public:
  TupleValue(const PTuple *ptype, std::shared_ptr<ArenaAllocator> region, char *p)
    : ptype(ptype),
      region(std::move(region)),
      p(p) {}

  const PTuple *get_ptype() const { return ptype; }
  size_t get_size() const { return ptype->element_ptypes.size(); }
  bool get_element_present(size_t i) const {
    return get_bit(p, i);
  }
  void set_element_present(size_t i, bool present) const {
    return set_bit(p, i, present);
  }
  inline Value get_element(size_t i) const;
  inline void set_element(size_t i, const Value &new_element) const;

  inline operator Value() const;
};

class Value {
  friend class StrValue;
  friend class ArrayValue;
  friend class TupleValue;

  const PType *ptype;
  bool present;
  std::shared_ptr<ArenaAllocator> region;
  union u {
    bool b;
    uint32_t i32;
    uint64_t i64;
    float f;
    double d;
    void *p;
  } u;

  void set_union_from(const union u &that_u) {
    switch (ptype->tag) {
    case PType::Tag::BOOL:
      u.b = that_u.b;
      break;
    case PType::Tag::INT32:
      u.i32 = that_u.i32;
      break;
    case PType::Tag::INT64:
      u.i64 = that_u.i64;
      break;
    case PType::Tag::FLOAT32:
      u.f = that_u.f;
      break;
    case PType::Tag::FLOAT64:
      u.d = that_u.d;
      break;
    case PType::Tag::STR:
    case PType::Tag::ARRAY:
    case PType::Tag::TUPLE:
      u.p = that_u.p;
      break;
    default:
      abort();
    }
  }

  /* creates a missing value of ptype `ptype` */
  Value(const PType *ptype)
    : ptype(ptype),
      present(false) {
  }
  Value(const PStr *ptype, std::shared_ptr<ArenaAllocator> region, void *p)
    : ptype(ptype),
      present(true),
      region(std::move(region)) {
    u.p = p;
  }
  Value(const PArray *ptype, std::shared_ptr<ArenaAllocator> region, void *p)
    : ptype(ptype),
      present(true),
      region(std::move(region)) {
    u.p = p;
  }
  Value(const PTuple *ptype, std::shared_ptr<ArenaAllocator> region, void *p)
    : ptype(ptype),
      present(true),
      region(std::move(region)) {
    u.p = p;
  }

public:
  Value(const PBool *ptype, bool b)
    : ptype(ptype),
      present(true) {
    u.b = b;
  }
  Value(const PInt32 *ptype, uint32_t i32)
    : ptype(ptype),
      present(true) {
    u.i32 = i32;
  }
  Value(const PInt64 *ptype, uint32_t i64)
    : ptype(ptype),
      present(true) {
    u.i64 = i64;
  }
  Value(const PFloat32 *ptype, float f)
    : ptype(ptype),
      present(true) {
    u.f = f;
  }
  Value(const PFloat64 *ptype, double d)
    : ptype(ptype),
      present(true) {
    u.d = d;
  }

  Value(const Value &that)
    : ptype(that.ptype),
      present(that.present),
      region(that.region) {
    if (present)
      set_union_from(that.u);
  }
  Value(const Value &&that)
    : ptype(that.ptype),
      present(that.present),
      region(std::move(that.region)) {
    if (present)
      set_union_from(that.u);
  }

  const Value &operator=(const Value &that) {
    ptype = that.ptype;
    region = that.region;
    set_union_from(that.u);
    return *this;
  }
  const Value &operator=(const Value &&that) {
    ptype = that.ptype;
    region = std::move(that.region);
    set_union_from(that.u);
    return *this;
  }

  static StrValue make_str(const PStr *ptype, std::shared_ptr<ArenaAllocator> region, size_t size);
  static ArrayValue make_array(const PArray *ptype, std::shared_ptr<ArenaAllocator> region, size_t size);
  static TupleValue make_tuple(const PTuple *ptype, std::shared_ptr<ArenaAllocator> region);

  static Value load(const PType *ptype, std::shared_ptr<ArenaAllocator> region, void *p);
  static void store(void *p, const Value &value);

  const PType *get_ptype() const { return ptype; }
  bool get_present() const { return present; }

  bool as_bool() const {
    assert(ptype->tag == PType::Tag::BOOL);
    return u.b;
  }
  uint32_t as_int32() const {
    assert(ptype->tag == PType::Tag::INT32);
    return u.i32;
  }
  uint64_t as_int64() const {
    assert(ptype->tag == PType::Tag::INT64);
    return u.i64;
  }
  float as_float32() const {
    assert(ptype->tag == PType::Tag::FLOAT32);
    return u.f;
  }
  double as_float64() const {
    assert(ptype->tag == PType::Tag::FLOAT64);
    return u.d;
  }
  StrValue as_str() const {
    assert(ptype->tag == PType::Tag::STR);
    return StrValue(cast<PStr>(ptype), region, (StrData *)u.p);
  }
  ArrayValue as_array() const {
    assert(ptype->tag == PType::Tag::ARRAY);
    return ArrayValue(cast<PArray>(ptype), region, (ArrayData *)u.p);
  }
  TupleValue as_tuple() const {
    assert(ptype->tag == PType::Tag::TUPLE);
    return TupleValue(cast<PTuple>(ptype), region, (char *)u.p);
  }
};

StrValue::operator Value() const {
  return Value(ptype, region, p);
}

ArrayValue::operator Value() const {
  return Value(ptype, region, p);
}

Value
ArrayValue::get_element(size_t i) const {
  return Value::load(ptype->element_ptype,
		     region,
		     p->elements + i * ptype->element_stride);
}

TupleValue::operator Value() const {
  return Value(ptype, region, p);
}

Value
TupleValue::get_element(size_t i) const {
  return Value::load(ptype->element_ptypes[i], region, p + ptype->element_offsets[i]);
}

void
TupleValue::set_element(size_t i, const Value &new_element) const {
  Value::store(p + ptype->element_offsets[i], new_element);
}

extern void format1(FormatStream &s, const Value &value);

}

#endif
