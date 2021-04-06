#ifndef HAIL_VALUE_HPP_INCLUDED
#define HAIL_VALUE_HPP_INCLUDED 1

#include <memory>

#include <hail/allocators.hpp>
#include <hail/tunion.hpp>
#include <hail/vtype.hpp>

namespace hail {

class FormatStream;
class Value;

struct StrData {
  uint32_t size;
  char data[1];
};

class StrValue {
  friend class Value;

public:
  const VStr *vtype;
private:
  std::shared_ptr<ArenaAllocator> region;
  StrData *p;

public:
  StrValue(const VStr *vtype, std::shared_ptr<ArenaAllocator> region, StrData *p)
    : vtype(vtype),
      region(std::move(region)),
      p(p) {}

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

public:
  const VArray *vtype;
private:
  std::shared_ptr<ArenaAllocator> region;
  ArrayData *p;
  size_t size;
  char *missing_bits;
  char *elements;

public:
  ArrayValue(const VArray *vtype, std::shared_ptr<ArenaAllocator> region, ArrayData *p)
    : vtype(vtype),
      region(std::move(region)),
      p(p),
      size(p->size),
      missing_bits(p->missing_bits),
      elements(p->elements) {}

  size_t get_size() const { return size; }
  bool get_element_missing(size_t i) const {
    return get_bit(missing_bits, i);
  }
  void set_element_missing(size_t i, bool missing) {
    set_bit(missing_bits, i, missing);
  }
  inline Value get_element(size_t i) const;
  inline void set_element(size_t i, const Value &new_element);

  inline operator Value() const;
};

class TupleValue {
  friend class Value;

public:
  const VTuple *vtype;
private:

  std::shared_ptr<ArenaAllocator> region;
  char *p;

public:
  TupleValue(const VTuple *vtype, std::shared_ptr<ArenaAllocator> region, char *p)
    : vtype(vtype),
      region(std::move(region)),
      p(p) {}

  size_t get_size() const { return vtype->element_vtypes.size(); }
  bool get_element_missing(size_t i) const {
    return get_bit(p, i);
  }
  void set_element_missing(size_t i, bool missing) const {
    return set_bit(p, i, missing);
  }
  inline Value get_element(size_t i) const;
  inline void set_element(size_t i, const Value &new_element) const;

  inline operator Value() const;
};

class Value {
  friend class StrValue;
  friend class ArrayValue;
  friend class TupleValue;
  friend void format1(FormatStream &s, const Value &value);

public:
  union PrimitiveUnion {
    bool b;
    int32_t i32;
    int64_t i64;
    float f;
    double d;
    void *p;
  };

  using PrimitiveUnion = union PrimitiveUnion;

  class Raw {
  public:
    bool missing;
    PrimitiveUnion u;

    Raw() {}
    Raw(const Value &value)
      : missing(value.missing),
	u(value.u) {}

    Raw &operator=(const Value &value) {
      missing = value.missing;
      u = value.u;
      return *this;
    }
  };

  static_assert(sizeof(Raw) == 16);
  static_assert(alignof(Raw) == 8);
  static_assert(offsetof(Raw, missing) == 0);
  static_assert(offsetof(Raw, u) == 8);

  const VType *vtype;
private:
  // FIXME do we want this?  Any values should be scoped inside a
  // region.
  std::shared_ptr<ArenaAllocator> region;
  bool missing;
  PrimitiveUnion u;

  void set_union_from(const PrimitiveUnion &that_u) {
    switch (vtype->tag) {
    case VType::Tag::BOOL:
      u.b = that_u.b;
      break;
    case VType::Tag::INT32:
      u.i32 = that_u.i32;
      break;
    case VType::Tag::INT64:
      u.i64 = that_u.i64;
      break;
    case VType::Tag::FLOAT32:
      u.f = that_u.f;
      break;
    case VType::Tag::FLOAT64:
      u.d = that_u.d;
      break;
    case VType::Tag::STR:
    case VType::Tag::ARRAY:
    case VType::Tag::TUPLE:
      u.p = that_u.p;
      break;
    default:
      abort();
    }
  }

  Value(const VStr *vtype, std::shared_ptr<ArenaAllocator> region, void *p)
    : vtype(vtype),
      region(std::move(region)),
      missing(false) {
    u.p = p;
  }
  Value(const VArray *vtype, std::shared_ptr<ArenaAllocator> region, void *p)
    : vtype(vtype),
      region(std::move(region)),
      missing(false) {
    u.p = p;
  }
  Value(const VTuple *vtype, std::shared_ptr<ArenaAllocator> region, void *p)
    : vtype(vtype),
      region(std::move(region)),
      missing(false) {
    u.p = p;
  }

public:
  /* creates a missing value of vtype `vtype` */
  Value(const VType *vtype)
    : vtype(vtype),
      missing(true) {
  }
  Value(const VBool *vtype, bool b)
    : vtype(vtype),
      missing(false) {
    u.b = b;
  }
  Value(const VInt32 *vtype, uint32_t i32)
    : vtype(vtype),
      missing(false) {
    u.i32 = i32;
  }
  Value(const VInt64 *vtype, uint32_t i64)
    : vtype(vtype),
      missing(false) {
    u.i64 = i64;
  }
  Value(const VFloat32 *vtype, float f)
    : vtype(vtype),
      missing(false) {
    u.f = f;
  }
  Value(const VFloat64 *vtype, double d)
    : vtype(vtype),
      missing(false) {
    u.d = d;
  }

  Value(const VType *vtype, std::shared_ptr<ArenaAllocator> region, Raw raw)
    : vtype(vtype),
      region(std::move(region)),
      missing(raw.missing),
      u(raw.u) {}

  Value(const Value &that)
    : vtype(that.vtype),
      missing(that.missing),
      region(that.region) {
    if (!missing)
      set_union_from(that.u);
  }
  Value(const Value &&that)
    : vtype(that.vtype),
      missing(that.missing),
      region(std::move(that.region)) {
    if (!missing)
      set_union_from(that.u);
  }

  const Value &operator=(const Value &that) {
    vtype = that.vtype;
    region = that.region;
    set_union_from(that.u);
    return *this;
  }
  const Value &operator=(const Value &&that) {
    vtype = that.vtype;
    region = std::move(that.region);
    set_union_from(that.u);
    return *this;
  }

  static StrValue make_str(const VStr *vtype, std::shared_ptr<ArenaAllocator> region, size_t size);
  static ArrayValue make_array(const VArray *vtype, std::shared_ptr<ArenaAllocator> region, size_t size);
  static TupleValue make_tuple(const VTuple *vtype, std::shared_ptr<ArenaAllocator> region);

  static Value load(const VType *vtype, std::shared_ptr<ArenaAllocator> region, void *p);
  static void store(void *p, const Value &value);

  bool get_missing() const { return missing; }

  bool as_bool() const {
    assert(vtype->tag == VType::Tag::BOOL);
    return u.b;
  }
  int32_t as_int32() const {
    assert(vtype->tag == VType::Tag::INT32);
    return u.i32;
  }
  int64_t as_int64() const {
    assert(vtype->tag == VType::Tag::INT64);
    return u.i64;
  }
  float as_float32() const {
    assert(vtype->tag == VType::Tag::FLOAT32);
    return u.f;
  }
  double as_float64() const {
    assert(vtype->tag == VType::Tag::FLOAT64);
    return u.d;
  }
  StrValue as_str() const {
    assert(vtype->tag == VType::Tag::STR);
    return StrValue(cast<VStr>(vtype), region, (StrData *)u.p);
  }
  ArrayValue as_array() const {
    assert(vtype->tag == VType::Tag::ARRAY);
    return ArrayValue(cast<VArray>(vtype), region, (ArrayData *)u.p);
  }
  TupleValue as_tuple() const {
    assert(vtype->tag == VType::Tag::TUPLE);
    return TupleValue(cast<VTuple>(vtype), region, (char *)u.p);
  }
};

StrValue::operator Value() const {
  return Value(vtype, region, p);
}

ArrayValue::operator Value() const {
  return Value(vtype, region, p);
}

Value
ArrayValue::get_element(size_t i) const {
  return Value::load(vtype->element_vtype,
		     region,
		     p->elements + i * vtype->element_stride);
}

void 
ArrayValue::set_element(size_t i, const Value &new_element) {
  Value::store(p->elements + i * vtype->element_stride, new_element);
}

TupleValue::operator Value() const {
  return Value(vtype, region, p);
}

Value
TupleValue::get_element(size_t i) const {
  if (get_element_missing(i))
    return Value(vtype->element_vtypes[i]);
  else
    return Value::load(vtype->element_vtypes[i], region, p + vtype->element_offsets[i]);
}

void
TupleValue::set_element(size_t i, const Value &new_element) const {
  auto missing = new_element.get_missing();
  set_element_missing(i, missing);
  if (!missing)
    Value::store(p + vtype->element_offsets[i], new_element);
}

void format1(FormatStream &s, const Value &value);

}

#endif
