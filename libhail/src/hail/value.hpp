class StrValue {
  const TStr *type;
  std::shared_ptr<ArenaAllocator> region;
  size_t size;
  void *data;

public:
  StrValue(TArray *type, std::shared_ptr<ArenaAllocator> region, void *p);

  size_t get_size() const { return size; }
  char *get_data() { return data; }
};

class ArrayValue {
  const TArray *type;
  std::shared_ptr<ArenaAllocator> region;
  size_t size;
  void *missing_bits;
  void *elements;

public:
  ArrayValue(TArray *type, std::shared_ptr<ArenaAllocator> region, void *p);

  size_t get_size() const;
  bool get_element_present(size_t i) const;
  void set_element_present(size_t i, bool present);
  Value get_element(size_t i) const;
  void set_element(size_t i, const Value &new_element);
};

class TupleValue {
public:
  TupleValue(const TTuple *type, std::shared_ptr<ArenaAllocator> region, void *p);

  bool get_element_present(size_t i) const;
  void set_element_present(size_t i) const;
  Value get_element(size_t i) const;
  void set_element(size_t i, const Value &new_element) const;
};

class Value {
  friend class StrValue;
  friend class ArrayValue;
  friend class TupleValue;

public:
  const Type *const type;
private:
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
    switch (tag) {
    case VOID:
      break
    case BOOL:
      u.b = that.u.b;
      break;
    case INT32:
      u.i32 = that.u.i32;
      break;
    case INT64:
      u.i64 = that.u.i64;
      break;
    case FLOAT32:
      u.f = u.f;
      break;
    case FLOAT64:
      u.d = u.d;
      break;
    case STR:
    case ARRAY:
    case TUPLE:
      u.p = u.p;
      break;
    default:
      abort();
    }
  }

  Value(const Type *type)
    : type(type),
      present(false) {
  }
  Value(const TStr *type, std::shared_ptr<ArenaAllocator> region, void *p)
    : type(type),
      present(present),
      region(std::move(region)) {
    u.p = p;
  }
  Value(const TArray *type, std::shared_ptr<ArenaAllocator> region, void *p)
    : type(type),
      present(true),
      region(std::move(region)) {
    u.p = p;
  }
  Value(const TTuple *type, std::shared_ptr<ArenaAllocator> region, void *p)
    : type(type),
      present(true),
      region(std::move(region)) {
    u.p = p;
  }

public:
  Value(TVoid *type)
    : type(type),
      present(true) {}
  Value(const Value &that)
    : type(that.type),
      present(that.present),
      region(that.region) {
    if (present)
      set_union_from(that.u);
  }
  Value(const Value &&that)
    : type(that.type),
      present(that.present),
      region(std::move(that.region)) {
    if (present)
      set_union_from(that.u);
  }
  Value(TBool *type, bool b)
    : type(type),
      present(true) {
    u.b = b;
  }
  Value(TInt32 *type, uint32_t i32)
    : type(type),
      present(true) {
    u.i32 = i32;
  }
  Value(TInt64 *type, uint32_t i64)
    : type(type),
      present(true) {
    u.i64 = i64;
  }
  Value(TFloat32 *type, float f)
    : type(type),
      present(true) {
    u.f = f;
  }
  Value(TFloat64 *type, double d)
    : type(type),
      present(true) {
    u.d = d;
  }

  static Value make_na(const Type *type) {
    return Value(ValueNAToken(), type);
  }
  static StrValue make_str(const TStr *type, std::shared_ptr<ArenaAllocator> region, size_t size);
  static ArrayValue make_array(const TArray *type, std::shared_ptr<ArenaAllocator> region, size_t size);
  static TupleValue make_tuple(const TTuple *type, std::shared_ptr<ArenaAllocator> region);

  const Value &operator=(const Value &that) {
    type = that.type;
    region = that.region;
    set_union_from(that.u);
  }
  const Value &operator=(const Value &&that) {
    type = that.type;
    region = std::move(that.region);
    set_union_from(that.u);
  }

  bool get_present() const { return present }

  bool as_bool() const {
    assert(type->tag == Type::Tag::BOOL);
    return u.b;
  }
  uint32_t as_int32() const {
    assert(type->tag == Type::Tag::INT32);
    return u.i32;
  }
  uint64_t as_int64() const {
    assert(type->tag == Type::Tag::INT64);
    return u.i64;
  }
  float as_float32() const {
    assert(type->tag == Type::Tag::FLOAT32);
    return u.f;
  }
  double as_float64() const {
    assert(type->tag == Type::Tag::FLOAT64);
    return u.d;
  }
  ArrayValue as_array() const {
    assert(type->tag == Type::Tag::ARRAY);
    return ArrayValue(type, region, u.p);
  }
};

extern void format1(FormatStream &s, const Value &value);
