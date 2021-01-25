#include <hail/format.hpp>
#include <hail/tunion.hpp>
#include <hail/type.hpp>

namespace hail {

Type::~Type() {}

void
format1(FormatStream &s, const Type *v) {
  switch (v->tag) {
  case Type::Tag::VOID:
    format(s, "void");
    break;
  case Type::Tag::BOOL:
    format(s, "bool");
    break;
  case Type::Tag::INT32:
    format(s, "int32");
    break;
  case Type::Tag::INT64:
    format(s, "int64");
    break;
  case Type::Tag::FLOAT32:
    format(s, "float32");
    break;
  case Type::Tag::FLOAT64:
    format(s, "float64");
    break;
  case Type::Tag::STR:
    format(s, "str");
    break;
  case Type::Tag::ARRAY:
    format(s, "array<", cast<TArray>(v)->element_type, ">");
    break;
  default:
    abort();
  }
}

TypeContext::TypeContext(HeapAllocator &heap)
  : arena(heap),
    tbool(arena.make<TBool>(TypeContextToken())),
    tint32(arena.make<TInt32>(TypeContextToken())),
    tint64(arena.make<TInt64>(TypeContextToken())) {}

const TArray *
TypeContext::tarray(const Type *element_type) {
  auto p = arrays.insert({element_type, nullptr});
  if (p.second) {
    TArray *t = arena.make<TArray>(TypeContextToken(), element_type);
    p.first->second = t;
    return t;
  } else
    return p.first->second;
}

}
