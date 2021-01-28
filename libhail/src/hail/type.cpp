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
  case Type::Tag::TUPLE:
    {
      format(s, "(");
      bool first = true;
      for (auto t : cast<TTuple>(v)->element_types) {
	if (first)
	  first = false;
	else
	  s.puts(", ");
	format(s, t);
      }
      format(s, ")");
    }
    break;
  default:
    abort();
  }
}

TypeContext::TypeContext(HeapAllocator &heap)
  : arena(heap),
    tvoid(arena.make<TVoid>(TypeContextToken())),
    tbool(arena.make<TBool>(TypeContextToken())),
    tint32(arena.make<TInt32>(TypeContextToken())),
    tint64(arena.make<TInt64>(TypeContextToken())),
    tfloat32(arena.make<TFloat32>(TypeContextToken())),
    tfloat64(arena.make<TFloat64>(TypeContextToken())),
    tstr(arena.make<TStr>(TypeContextToken())) {}

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

const TStream *
TypeContext::tstream(const Type *element_type) {
  auto p = streams.insert({element_type, nullptr});
  if (p.second) {
    TStream *t = arena.make<TStream>(TypeContextToken(), element_type);
    p.first->second = t;
    return t;
  } else
    return p.first->second;
}

const TTuple *
TypeContext::ttuple(const std::vector<const Type *> &element_types) {
  /* it is not clear how to insert into tuples without copying
     element_types unnecessarily or searching tuples twice */
  auto i = tuples.find(element_types);
  if (i != tuples.end())
    return i->second;
  TTuple *t = arena.make<TTuple>(TypeContextToken(), element_types);
  tuples.insert({element_types, t});
  return t;
}

}
