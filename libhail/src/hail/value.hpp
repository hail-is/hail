
class ArrayValue {
  const TArray *type;
  uint64_t length;
  void *missing_bits;
  void *elements;
};

class Value {
  const Type *type;
  union {
    bool b;
    uint32_t i32;
    uint64_t i64;
    float f;
    double d;
    void *p;
  } u;

  bool as_bool() const {
    assert(type->tag == Type::Tag::BOOL);
    return u.b;
  }
  uint32_t as_int32() const;
  uint64_t as_int64() const;
  float as_float() const;
  double as_double() const;
  ArrayValue as_array() const;
};
