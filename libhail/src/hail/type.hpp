#ifndef HAIL_TYPE_HPP
#define HAIL_TYPE_HPP 1

#include <map>

#include <hail/allocators.hpp>

namespace hail {

class TypeContext;

class TypeContextToken {
  friend class TypeContext;
  TypeContextToken() {}
};

class Type {
public:
  using BaseType = Type;
  enum class Tag {
    VOID,
    BOOL,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STR,
    ARRAY,
    STREAM,
    TUPLE
  };
  const Tag tag;
  Type(Tag tag) : tag(tag) {}
  virtual ~Type();
};

void format1(FormatStream &s, const Type *v);

class TBool : public Type {
public:
  static const Tag self_tag = Type::Tag::BOOL;
  TBool(TypeContextToken) : Type(self_tag) {}
};

class TInt32 : public Type {
public:
  static const Tag self_tag = Type::Tag::INT32;
  TInt32(TypeContextToken) : Type(self_tag) {}
};

class TInt64 : public Type {
public:
  static const Tag self_tag = Type::Tag::INT64;
  TInt64(TypeContextToken) : Type(self_tag) {}
};

class TArray : public Type {
public:
  static const Tag self_tag = Type::Tag::ARRAY;
  const Type *const element_type;
  TArray(TypeContextToken, const Type *element_type) : Type(self_tag), element_type(element_type) {}
};

class TypeContext {
  ArenaAllocator<HeapAllocator> arena;

  std::map<const Type *, const TArray *> arrays;

public:
  const TBool *const tbool;
  const TInt32 *const tint32;
  const TInt64 *const tint64;

  TypeContext(HeapAllocator &heap);

  const TArray *tarray(const Type *element_type);
};

}

#endif
